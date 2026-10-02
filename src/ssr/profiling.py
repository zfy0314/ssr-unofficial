"""Trajectory-wide accounting with every turn retained, including failed tools.

Every turn preserves candidate scores/probabilities, context length, tool status,
raw engine metadata (including reported cached-token counts), and phase timing.
The trajectory aggregates all turns and retains initial setup/system prefill.
The run profile retains every example/sample, adds their totals and records engine
load time separately. Context length is a snapshot, not newly generated tokens.

Token meanings:
- total_generated_tokens: newly decoded action tokens only, including returned
  stop tokens; injected reasoning is not generated.
- thinking_tokens: selected reasoning body; generated_thinking_tokens excludes
  injection. total_response_tokens includes injected reasoning and delimiters.
- selected_candidate_tokens: full wrapped selected candidate.
- candidate_tokens_scored: work summed over the entire candidate library.
- observation_tokens: tool response and next-assistant framing tokens.

Timing scopes are nested, not additive across every field:
- total_generation_seconds already includes shared prefill, scoring setup,
  reasoning scoring/selection and action generation. Do not add components twice.
- model_generation_with_system_prefill_seconds adds the initial system prefill
  once to the generation total.
- trajectory_wall_seconds includes initial input/context setup and inference
  (including completed-turn serialization), excluding model/processor loading
  and final report serialization. Media preparation and tools are outside model
  generation timing. Engine load is reported separately at run level.

These are live diagnostics, not isolated CUDA kernel timings or benchmark claims.
Compare matched models, inputs, cache states, tools, stopping and hardware.
"""
from collections import Counter
import math

TIME_FIELDS = ('media_preparation_seconds', 'prefill_seconds', 'scoring_setup_seconds',
               'thinking_only_seconds', 'action_generation_seconds', 'total_generation_seconds',
               'tool_seconds', 'orchestration_seconds', 'turn_wall_seconds')
TOKEN_FIELDS = ('total_generated_tokens', 'total_response_tokens', 'thinking_tokens',
                'generated_thinking_tokens', 'selected_candidate_tokens',
                'candidate_tokens_scored', 'observation_tokens')


def trajectory_profile(result):
    turns = result['turns']
    totals = {key: math.fsum(t.get(key, 0.) for t in turns) for key in TIME_FIELDS}
    totals.update({key: sum(t.get(key, 0) for t in turns) for key in TOKEN_FIELDS})
    totals.update(system_prefill_seconds=result.get('system_prefill_seconds', 0.),
                  setup_seconds=result.get('setup_seconds', 0.),
                  elapsed_wall_seconds=result.get('elapsed_wall_seconds', result.get('inference_wall_seconds', 0.)),
                  serpapi_requests=result.get('serpapi_requests', 0))
    totals['trajectory_wall_seconds'] = totals['setup_seconds'] + totals['elapsed_wall_seconds']
    totals['model_generation_with_system_prefill_seconds'] = totals['system_prefill_seconds'] + totals['total_generation_seconds']
    records = []
    for t in turns:
        records.append({k:v for k,v in t.items() if k not in ('h','z','a','o','reasoning','action')})
    return dict(schema_version=1, status=result['delta'], completed_turns=len(turns),
                totals=totals, per_turn=records,
                tool_status_counts=dict(Counter(t.get('tool_result', {}).get('status', 'ok')
                     for t in turns if t.get('parsed', {}).get('kind') == 'tool')),
                final_context_tokens=len(result['h']))


def run_profile(profiles, engine_load_seconds):
    """Retain each trajectory and sum its complete statistics without overwriting."""
    fields = list(TIME_FIELDS) + list(TOKEN_FIELDS) + [
        'system_prefill_seconds', 'setup_seconds', 'elapsed_wall_seconds',
        'trajectory_wall_seconds', 'model_generation_with_system_prefill_seconds', 'serpapi_requests']
    totals = {key: sum(p['totals'][key] for p in profiles) for key in fields}
    return dict(schema_version=1, trajectories_completed=len(profiles),
                completed_turns=sum(p['completed_turns'] for p in profiles),
                engine_load_seconds=engine_load_seconds, totals=totals,
                status_counts=dict(Counter(p['status'] for p in profiles)), trajectories=profiles)
