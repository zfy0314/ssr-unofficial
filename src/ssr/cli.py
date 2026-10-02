"""Portable SSR inference command; credentials are read only from environment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

from .inference import SSR
from .profiling import trajectory_profile, run_profile
from .library import asset, load_library, render_system_prompt, tokenize_library


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Qwen3-VL Hugging Face ID or local checkpoint directory")
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path)
    inputs.add_argument("--input-jsonl", type=Path, help="Rows: id, image, question; image paths relative to this file")
    p.add_argument("--question", "--prompt", help="Question for --image")
    p.add_argument("--library", type=Path)
    p.add_argument("--system-prompt", type=Path, help="Override the complete system prompt")
    p.add_argument("--alpha", type=float, default=1., help="Length normalization exponent, [0,1]")
    p.add_argument("--tau", type=float, default=0., help="Reasoning selection temperature; 0 selects first maximum")
    p.add_argument("--temperature", type=float, default=0., help="Action token sampling temperature (independent of tau)")
    p.add_argument("--top-p", type=float, default=1.)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tools", choices=["none", "local", "live"], default="live",
                   help="live (default): SerpAPI text search, recorded image search and crop; local: crop only; none: emit only")
    p.add_argument("--image-results", type=Path, help="Cached image-search manifest; defaults to the bundled teaser results")
    p.add_argument("--search-results", type=int, default=5)
    p.add_argument("--summarizer-url", help="Optional chat API base URL ending in /v1 for whole-page summaries")
    p.add_argument("--summarizer-model", default="qwen3-32b")
    p.add_argument("--cache", type=Path, default=Path(".cache/ssr"))
    p.add_argument("--output", type=Path, default=Path("outputs/run"), help="New or empty directory")
    p.add_argument("--context-length", type=int, default=32768)
    p.add_argument("--mem-fraction-static", type=float, default=.7)
    p.add_argument("--vlm-cache-size-mb", type=int, default=4096)
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--attention-backend", help="Optional SGLang attention backend override")
    p.add_argument("--offline", action="store_true", help="Use only cached/local model files")
    p.add_argument("--profiling", action="store_true", help="Print timing totals; per-turn timings are always saved")
    p.add_argument("--dry-run", action="store_true", help="Validate files, library and options without weights, GPU or API calls")
    return p


def main(argv=None):
    """Run one persistent engine over all input rows and repeated samples.

    Inputs: --image/--question or --input-jsonl. JSONL rows contain image and
    question plus optional id; image paths resolve relative to the JSONL file.
    A local checkpoint works with --offline; otherwise HF snapshot_download
    resolves/downloads the model. Source the desired environment first so ninja
    is available for FlashInfer's possible first-run CUDA kernel compilation.

    Outputs: settings.json records arguments, model/package versions, candidate
    token IDs and source hashes. Each example/sample directory has turns.jsonl
    (appended after each completed turn), trajectory.json (full context, IDs,
    scores, observations, media provenance and status), and profiling.json.
    Root profiling.json retains all completed trajectories and their totals.
    Completed JSONL records survive interruption before final report writing.
    Credentials come only from environment, never automatically from .env or
    shell startup files. CPU validation: install .[test], then python -m pytest
    -q. --dry-run needs neither CUDA nor model weights nor API requests.
    """
    p = parser()
    args = p.parse_args(argv)
    if not all(math.isfinite(x) for x in [args.alpha, args.tau, args.temperature, args.top_p, args.mem_fraction_static]):
        p.error("Numeric parameters must be finite")
    if not (0 <= args.alpha <= 1 and args.tau >= 0 and args.temperature >= 0 and 0 < args.top_p <= 1
            and 0 < args.mem_fraction_static < 1 and args.max_turns > 0 and args.num_samples > 0
            and args.max_new_tokens > 0 and args.search_results > 0 and args.vlm_cache_size_mb > 0
            and args.context_length > args.max_new_tokens and 0 < args.port < 65536
            and (args.top_k == -1 or args.top_k > 0)):
        p.error("Invalid inference parameters")
    entries = load_library(args.library)
    system = args.system_prompt.read_text() if args.system_prompt else render_system_prompt(entries)
    if args.image:
        if not args.question or not args.question.strip():
            p.error("--image requires --question")
        rows = [{"id": "example", "image": str(args.image.resolve()), "question": args.question}]
    else:
        rows = [json.loads(line) for line in args.input_jsonl.read_text().splitlines() if line.strip()]
        for row in rows:
            row["image"] = str((args.input_jsonl.resolve().parent / row["image"]).resolve())
    if not rows:
        p.error("No input examples")
    for row in rows:
        if not isinstance(row.get("question"), str) or not row["question"].strip() or not Path(row["image"]).is_file():
            p.error("Every input must contain an existing image and nonempty question")
    if args.dry_run:
        print(json.dumps({"status": "dry-run-ok", "examples": len(rows), "candidates": len(entries),
                          "alpha": args.alpha, "tau": args.tau, "tools": args.tools,
                          "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest()}, indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Output directory must be new or empty")
    if args.tools == "live" and not (os.environ.get("SERPAPI_API_KEY") or os.environ.get("SERP_API_KEY")):
        p.error("--tools live requires SERPAPI_API_KEY (or SERP_API_KEY)")
    # Delayed imports keep CPU dry-run/tests independent of CUDA installation.
    from huggingface_hub import snapshot_download
    from .backend import SGLangEngine
    from .harness import SearchHarness
    from .hashing import atomic_write_json
    from .media import MediaObjectCache
    from .token_counts import response_token_counts
    from .tools import SerpTools
    model = str(Path(args.model).resolve()) if Path(args.model).is_dir() else snapshot_download(args.model, local_files_only=args.offline)
    args.output.mkdir(parents=True, exist_ok=True)
    media_cache = MediaObjectCache(args.cache / "media", model)
    tokenizer = media_cache.processor.tokenizer
    R = tokenize_library(entries, tokenizer)
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eta = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
               top_p=args.top_p, top_k=args.top_k, stop_token_ids=[eos], skip_special_tokens=False)
    tools = SerpTools(args.cache / "tools", top_k=args.search_results,
                     summarizer_url=args.summarizer_url, summarizer_model=args.summarizer_model, image_results=args.image_results)
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest.update(model_resolved=model, eta=eta, candidates=[dict(id=key, token_ids=r, **entry)
                     for (key, entry), r in zip(entries.items(), R, strict=True)],
                    system_prompt_sha256=hashlib.sha256(system.encode()).hexdigest(),
                    packages={k:importlib.metadata.version(k) for k in ["sglang", "torch", "transformers"]},
                    source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")},
                    provenance=json.loads(asset("provenance.json").read_text()))
    atomic_write_json(args.output / "settings.json", manifest)
    (args.output / "system_prompt.txt").write_text(system)
    engine_start = time.perf_counter()
    engine = SGLangEngine(model, context_length=args.context_length, mem_fraction_static=args.mem_fraction_static,
                          vlm_cache_size_mb=args.vlm_cache_size_mb, seed=args.seed, port=args.port,
                          attention_backend=args.attention_backend)
    engine_load_seconds = time.perf_counter()-engine_start
    profiles = []
    try:
        atomic_write_json(args.output / "engine_compatibility.json", engine.compatibility)
        for row_index, row in enumerate(rows):
            for sample in range(args.num_samples):
                # Fresh per-rollout media bookkeeping; engine retains its caches.
                # Supplying pixels again is safe if an earlier image was evicted.
                setup_start = time.perf_counter()
                requests_before = tools.requests_made
                H = SearchHarness(tokenizer, media_cache, Path(row["image"]), tools,
                                  mode=args.tools)
                s, h = H.initial_context(system, row["question"])
                output = args.output / f"example_{row_index+1:04}_sample_{sample+1:03}"
                output.mkdir()
                seed = args.seed + row_index*args.num_samples + sample
                controller = SSR(engine, R, alpha=args.alpha, tau=args.tau, eta=eta, T_max=args.max_turns, seed=seed)
                def save_turn(turn):
                    turn["candidate_id"] = list(entries)[turn["index"]-1]
                    turn["reasoning"] = tokenizer.decode(turn["z"], skip_special_tokens=False)
                    turn["action"] = tokenizer.decode(turn["a"], skip_special_tokens=False)
                    turn.update(response_token_counts(tokenizer, turn["z"]+turn["a"], len(turn["z"])))
                    with (output / "turns.jsonl").open("a") as stream:
                        stream.write(json.dumps(turn, ensure_ascii=False) + "\n")
                    print(f"Turn {turn['turn']+1}: {turn['candidate_id']}\n{turn['reasoning']}\n{turn['action']}", flush=True)
                    if "tool_result" in turn:
                        print(f"Tool status: {turn['tool_result'].get('status', 'ok')} | Turn status: {turn['delta']}", flush=True)
                start = time.perf_counter()
                setup_seconds = start-setup_start
                result = controller.infer(s, h, H, on_turn=save_turn)
                result.update(input=row, sample=sample+1, selection_seed=seed,
                              elapsed_wall_seconds=time.perf_counter()-start, setup_seconds=setup_seconds,
                              serpapi_requests=tools.requests_made-requests_before,
                              context_text=tokenizer.decode(result["h"], skip_special_tokens=False),
                              images=[a.manifest_record() for a in H.registry], serpapi_requests_cumulative=tools.requests_made)
                profile = trajectory_profile(result)
                result["summary"] = profile["totals"]
                atomic_write_json(output / "profiling.json", profile)
                profiles.append(dict(example_index=row_index+1, sample=sample+1,
                                     output=str(output), **profile))
                atomic_write_json(args.output / "profiling.json", run_profile(profiles, engine_load_seconds))
                atomic_write_json(output / "trajectory.json", result)
                print(f"Status: {result['delta']} | Saved: {output}", flush=True)
                if args.profiling:
                    print(json.dumps(result["summary"], indent=2), flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
