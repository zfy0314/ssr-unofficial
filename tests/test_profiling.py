import pytest
from ssr.profiling import TIME_FIELDS, TOKEN_FIELDS, trajectory_profile, run_profile


def test_all_turns_and_all_trajectories_are_aggregated():
    turns=[]
    for i in range(1,5):
        turn={key:float(i) for key in TIME_FIELDS}
        turn.update({key:i for key in TOKEN_FIELDS})
        turn.update(turn=i-1, context_tokens=i*10, scores=[-.1*i],
                    parsed={'kind':'tool'}, tool_result={'status':'tool-disabled'},
                    prefill_meta={'cached_tokens':i}, candidate_meta=[{'cached_tokens':i}],
                    action_meta={'completion_tokens':i})
        turns.append(turn)
    result=dict(turns=turns, h=[1]*50, delta='turn-limit', system_prefill_seconds=.2,
                setup_seconds=.3, elapsed_wall_seconds=25., serpapi_requests=0)
    profile=trajectory_profile(result)
    assert profile['completed_turns'] == 4
    assert all(profile['totals'][key] == 10 for key in TIME_FIELDS+TOKEN_FIELDS)
    assert profile['tool_status_counts'] == {'tool-disabled':4}
    assert profile['totals']['trajectory_wall_seconds'] == pytest.approx(25.3)
    assert profile['totals']['model_generation_with_system_prefill_seconds'] == pytest.approx(10.2)
    assert profile['per_turn'][3]['action_meta'] == {'completion_tokens':4}
    combined=run_profile([profile,profile],1.5)
    assert combined['completed_turns'] == 8
    assert combined['totals']['thinking_tokens'] == 20
    assert combined['engine_load_seconds'] == 1.5
    assert len(combined['trajectories']) == 2


def test_profile_empty_context_limit():
    p=trajectory_profile(dict(turns=[],h=[1],delta='context-limit'))
    assert p['totals']['total_generated_tokens'] == 0
    assert p['completed_turns'] == 0
