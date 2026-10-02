import json
import math
import random

import pytest

from ssr.inference import SSR, candidate_scores, select_reasoning
from ssr.library import asset, load_library, render_system_prompt


def test_prompt_and_library_match_synchronized_source():
    import hashlib
    entries = load_library()
    provenance = json.loads(asset("provenance.json").read_text())
    assert len(entries) == 6
    assert hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest() == provenance["library_entries_sha256"]
    assert hashlib.sha256(render_system_prompt(entries).encode()).hexdigest() == provenance["reference_system_prompt_sha256"]


def test_selection_mean_normalization_and_first_tie():
    R = [[10], [11, 12]]
    Y = [{"meta_info": {"input_token_logprobs": [(None, 99, ""), (-.4, 10, "")]}},
         {"meta_info": {"input_token_logprobs": [(None, 99, ""), (-.3, 11, ""), (-.3, 12, "")]}}]
    S, ell = candidate_scores(Y, R, 1.)
    assert ell == [[-.4], [-.3, -.3]]
    assert select_reasoning(S, 0., random.Random())[0] == 1
    assert select_reasoning([1., 1.], 0., random.Random())[0] == 0
    assert candidate_scores(Y, R, 0.)[0] == [-.4, -.6]


@pytest.mark.parametrize("row", [[None, 10], [float('nan'), 10], [-.1, 11]])
def test_rejects_misaligned_or_nonfinite_logprobs(row):
    with pytest.raises(ValueError):
        candidate_scores([{"meta_info": {"input_token_logprobs": [row]}}], [[10]], 1.)


def test_stable_categorical_and_seed():
    a = select_reasoning([-10000., -10001.], .7, random.Random(42))
    b = select_reasoning([-10000., -10001.], .7, random.Random(42))
    assert a == b
    assert math.isclose(sum(a[1]), 1.)
    assert a[1][0] > a[1][1]


class Engine:
    context_length = 1000

    def __init__(self, actions):
        self.actions = iter(actions)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("return_logprob"):
            boundary = kwargs["logprob_start_len"]
            return [{"meta_info": {"input_token_logprobs": [(None, x[boundary], "")] +
                     [(-.2-i, token, "") for token in x[boundary+1:]], "completion_tokens": 0}}
                    for i, x in enumerate(kwargs["input_ids"])]
        if kwargs["sampling_params"]["max_new_tokens"] == 0:
            return {"output_ids": []}
        return {"output_ids": next(self.actions), "meta_info": {"completion_tokens": 1}}


class Harness:
    last_tool_result = {"observation": "fake tool result"}

    def prepare_media(self): return {"format": "full"}
    def cached_media(self): return {"format": "reference"}
    def mark_media_cached(self): pass
    def parse(self, ids):
        return {"kind": {70:"tool", 80:"answer", 81:"invalid"}[ids[0]]}
    def execute(self, parsed): return [90, 91], "continue"


def test_full_two_turn_history_and_media_batch():
    engine = Engine([[70], [80]])
    result = SSR(engine, [[10, 11], [12]], eta={"max_new_tokens": 10}, T_max=3).infer([1], [1, 2], Harness())
    assert result["delta"] == "answer"
    assert result["h"] == [1, 2, 10, 11, 70, 90, 91, 10, 11, 80]
    assert len(result["turns"]) == 2
    assert result["turns"][0]["index"] == 1
    scoring = [c for c in engine.calls if c.get("return_logprob")]
    assert scoring[0]["logprob_start_len"] == 1
    assert len(scoring[0]["image_data"]) == 2
    assert scoring[1]["input_ids"][0] == result["turns"][1]["h"] + [10, 11]


@pytest.mark.parametrize("actions,limit,expected", [([[81]], 2, "invalid-output"), ([[70]], 1, "turn-limit")])
def test_stopping_statuses(actions, limit, expected):
    result = SSR(Engine(actions), [[10]], eta={"max_new_tokens": 10}, T_max=limit).infer([1], [1, 2], Harness())
    assert result["delta"] == expected
    if expected == "turn-limit":
        assert result["h"][-2:] == [90, 91]  # final-turn tool still executes


def test_tool_error_stops_after_appending_observation():
    class FailedHarness(Harness):
        def execute(self, parsed): return [92], "tool-error"
    result = SSR(Engine([[70]]), [[10]], eta={"max_new_tokens": 10}).infer([1], [1, 2], FailedHarness())
    assert result["delta"] == "tool-error"
    assert result["h"][-1] == 92


def test_context_limit_is_explicit():
    engine = Engine([])
    engine.context_length = 3
    result = SSR(engine, [[10]], eta={"max_new_tokens": 10}).infer([1], [1, 2], Harness())
    assert result["delta"] == "context-limit"
    assert result["turns"] == []


def test_four_turns_retain_every_disabled_tool_and_observation():
    class RecoverableHarness(Harness):
        def execute(self, parsed):
            self.last_tool_result = {'status':'tool-disabled', 'observation':'Use another tool'}
            return [92], 'continue'
    result = SSR(Engine([[70]]*4), [[10]], eta={'max_new_tokens':10}, T_max=4).infer([1], [1,2], RecoverableHarness())
    assert result['delta'] == 'turn-limit'
    assert len(result['turns']) == 4
    assert [t['delta'] for t in result['turns']] == ['continue']*3+['turn-limit']
    assert result['h'].count(92) == 4
    assert all(t['tool_result']['status'] == 'tool-disabled' for t in result['turns'])
    assert all(t['turn_wall_seconds'] >= t['total_generation_seconds'] for t in result['turns'])
