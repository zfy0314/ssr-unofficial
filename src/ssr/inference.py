"""Algorithm 1: SSR inference with parallel reasoning decoding.

Names follow the project webpage: h, R, K, X, Y, S, i_t, z_t, u_t, a_t, o_t,
delta, alpha, tau, eta and T_max. Candidate indices in records are one-based.
The engine owns the KV cache; no KV tensors cross this interface.

Algorithm mapping and numerical semantics:
    s is the system prefix; H.initial_context constructs h_0 = s || q.
    R contains complete wrapped candidates; K_i counts their delimiter tokens too.
    Warm s with g_0 (max_new_tokens=0), then explicitly prefill shared h_t to
    install new image embeddings. Compact media references are reused thereafter.
    X_i = h_t || r_i; a variable-length batched prefill returns Y with likelihoods
    starting at L_t-1. The tail K_i records must match the candidate token IDs.
    S_i = sum(log p(r_i,j | h_t, r_i,<j)) / K_i**alpha. alpha=1 is the mean;
    alpha=0 is the sum. Candidates use causal attention, never cross-attend,
    and their known token positions are scored in parallel rather than decoded.
    tau=0 selects the first maximum in library insertion order; tau>0 samples
    softmax(S/tau). Inject the full selected z_t and decode a_t using eta.
    Selection RNG is per rollout; engine action RNG is seeded once and depends
    on request order. Cross-device bitwise reproducibility is not guaranteed.

H validates the answer/tool schema and returns observation tokens including the
next assistant prefix. T_max is an upper bound: a final answer stops early. The
last permitted turn still executes and records its tool. Crop-only mode returns
recoverable unavailable-tool feedback; emit-only mode stops on a tool request.
Invalid outputs, actual tool failures and context exhaustion remain terminal.
"answer" is a syntactic stopping status, not a factual-correctness judgment.

"""
from __future__ import annotations

import math
import random
import time
from typing import Any


def select_reasoning(S: list[float], tau: float, rng: random.Random) -> tuple[int, list[float]]:
    """Return zero-based index internally; stable first maximum when tau=0."""
    if not S or not all(math.isfinite(s) for s in S) or not math.isfinite(tau) or tau < 0:
        raise ValueError("Finite scores and a finite nonnegative tau are required")
    if tau == 0:
        index = max(range(len(S)), key=S.__getitem__)
        return index, [float(i == index) for i in range(len(S))]
    weights = [math.exp((s - max(S)) / tau) for s in S]
    probabilities = [w / math.fsum(weights) for w in weights]
    return rng.choices(range(len(S)), weights=probabilities, k=1)[0], probabilities


def candidate_scores(Y: list[dict], R: list[list[int]], alpha: float) -> tuple[list[float], list[list[float]]]:
    """Check every candidate token ID before reducing its aligned log likelihoods."""
    if len(Y) != len(R) or not 0 <= alpha <= 1:
        raise ValueError("Candidate responses/normalization exponent are invalid")
    S, logprobs = [], []
    for response, r_i in zip(Y, R, strict=True):
        K_i = len(r_i)
        rows = response.get("meta_info", {}).get("input_token_logprobs")
        if not K_i or not isinstance(rows, list) or len(rows) < K_i:
            raise ValueError("Missing candidate input-token log probabilities")
        D_i = rows[-K_i:]
        if any(not isinstance(d, (list, tuple)) or len(d) < 2 for d in D_i):
            raise ValueError("Malformed input-token log-probability record")
        if [d[1] for d in D_i] != r_i:
            raise ValueError("Candidate token IDs do not align with returned log probabilities")
        if any(d[0] is None for d in D_i):
            raise ValueError("Null candidate log probability (possible off-by-one boundary)")
        ell_i = [float(d[0]) for d in D_i]
        if not all(math.isfinite(x) for x in ell_i):
            raise ValueError("Nonfinite candidate log probability")
        if response.get("output_ids") or response.get("meta_info", {}).get("completion_tokens", 0):
            raise ValueError("Prefill-only scoring unexpectedly generated tokens")
        S.append(math.fsum(ell_i) / K_i**alpha)
        logprobs.append(ell_i)
    return S, logprobs


class SSR:
    """Reusable controller over an SGLang-compatible engine and tool harness."""

    def __init__(self, pi_theta, R, *, alpha=1.0, tau=0.0, eta=None, T_max=8, seed=42):
        if not R or any(not r for r in R):
            raise ValueError("The reasoning library must contain nonempty token sequences")
        if not math.isfinite(alpha) or not 0 <= alpha <= 1 or not math.isfinite(tau) or tau < 0:
            raise ValueError("Require alpha in [0,1] and finite tau >= 0")
        if T_max < 1:
            raise ValueError("T_max must be positive")
        self.pi_theta, self.R = pi_theta, R
        self.alpha, self.tau, self.T_max = alpha, tau, T_max
        self.eta = dict(eta or {"max_new_tokens": 8192, "temperature": 0.0})
        if self.eta.get("max_new_tokens", 0) < 1 or self.eta.get("n", 1) != 1:
            raise ValueError("Action generation requires positive max_new_tokens and n=1")
        self.rng = random.Random(seed)

    def infer(self, s: list[int], h: list[int], H, *, on_turn=None) -> dict[str, Any]:
        """Return h and delta, plus auditable turns; H owns media and observations.

        A separate shared-context prefill makes the timing split explicit and
        installs new image embeddings before the candidate batch references them.
        It leaves Algorithm 1's scores and selection rule unchanged.
        """
        pi_theta, R, eta = self.pi_theta, self.R, self.eta
        K = [len(r) for r in R]
        g_0 = {"max_new_tokens": 0, "temperature": 0.0}
        h = list(h)
        turns = []
        warm_start = time.perf_counter()
        pi_theta.generate(input_ids=s, sampling_params=g_0, return_logprob=False)
        system_prefill_seconds = time.perf_counter() - warm_start
        delta = "turn-limit"
        for t in range(self.T_max):
            L_t = len(h)
            if L_t + max(K) + eta["max_new_tokens"] > pi_theta.context_length:
                delta = "context-limit"
                break
            turn_start = time.perf_counter()
            full_media = H.prepare_media()
            start = time.perf_counter()
            prefill = pi_theta.generate(input_ids=h, sampling_params=g_0,
                                       return_logprob=False, image_data=full_media)
            if prefill.get("output_ids"):
                raise ValueError("Shared prefill generated tokens")
            prefill_end = time.perf_counter()
            H.mark_media_cached()
            media = H.cached_media()
            X = [h + r_i for r_i in R]
            thinking_start = time.perf_counter()
            Y = pi_theta.generate(input_ids=X, sampling_params=g_0, return_logprob=True,
                                  logprob_start_len=L_t - 1,
                                  image_data=[media for _ in R] if media is not None else None)
            S, ell = candidate_scores(Y, R, self.alpha)
            index, probabilities = select_reasoning(S, self.tau, self.rng)
            i_t = index + 1
            z_t = R[index]
            u_t = h + z_t
            thinking_end = time.perf_counter()
            A = pi_theta.generate(input_ids=u_t, sampling_params=eta, return_logprob=False, image_data=media)
            action_end = time.perf_counter()
            a_t = list(A.get("output_ids") or [])
            completion = A.get("meta_info", {}).get("completion_tokens")
            if completion is not None and completion != len(a_t):
                raise ValueError("Engine completion count differs from returned action token IDs")
            h_plus = u_t + a_t
            parsed = H.parse(a_t)
            record = dict(turn=t, h=h, context_tokens=L_t,
                          media_preparation_seconds=start-turn_start,
                          scoring_setup_seconds=thinking_start-prefill_end,
                          prefill_meta=prefill.get("meta_info", {}),
                          candidate_meta=[{k:v for k,v in y.get("meta_info", {}).items()
                                           if k != "input_token_logprobs"} for y in Y], index=i_t, z=z_t, a=a_t, scores=S,
                          candidate_token_logprobs=ell, selection_probabilities=probabilities,
                          selected_candidate_logprob=math.log(probabilities[index]),
                          prefill_seconds=prefill_end-start,
                          thinking_only_seconds=thinking_end-thinking_start,
                          action_generation_seconds=action_end-thinking_end,
                          total_generation_seconds=action_end-start,
                          total_generated_tokens=len(a_t), total_response_tokens=len(z_t)+len(a_t),
                          candidate_tokens_scored=sum(K), selected_candidate_tokens=len(z_t),
                          action_meta=A.get("meta_info", {}), parsed=parsed, tool_seconds=0.0,
                          observation_tokens=0)
            h = h_plus
            if parsed["kind"] == "answer":
                delta = "answer"
            elif parsed["kind"] != "tool":
                delta = "invalid-output"
            else:
                tool_start = time.perf_counter()
                o_t, delta_t = H.execute(parsed)
                record["tool_seconds"] = time.perf_counter() - tool_start
                record["o"] = o_t
                record["observation_tokens"] = len(o_t)
                record["tool_result"] = H.last_tool_result
                h = h_plus + o_t
                delta = ("continue" if t+1 < self.T_max else "turn-limit") if delta_t == "continue" else delta_t
            record["delta"] = delta
            record["turn_wall_seconds"] = time.perf_counter()-turn_start
            record["orchestration_seconds"] = max(0., record["turn_wall_seconds"] -
                record["media_preparation_seconds"] - record["total_generation_seconds"] - record["tool_seconds"])
            turns.append(record)
            if on_turn:
                on_turn(record)
            if delta != "continue":
                break
        return dict(h=h, delta=delta, turns=turns, system_prefill_seconds=system_prefill_seconds,
                    inference_wall_seconds=time.perf_counter()-warm_start)
