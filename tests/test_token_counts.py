import pytest
from ssr.token_counts import response_token_counts


class CharacterTokenizer:
    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


def test_injected_reasoning_is_not_generated():
    z = '<thinking>two words</thinking>'
    a = '<answer>blue</answer>'
    counts = response_token_counts(CharacterTokenizer(), list(map(ord, z+a)), len(z))
    assert counts['total_generated_tokens'] == len(a)
    assert counts['thinking_tokens'] == len('two words')
    assert counts['generated_thinking_tokens'] == 0


def test_generated_reasoning_and_missing_span():
    raw = '<reason>abc</reason><answer>yes</answer>'
    counts = response_token_counts(CharacterTokenizer(), list(map(ord, raw)))
    assert counts['thinking_tokens'] == counts['generated_thinking_tokens'] == 3
    assert not response_token_counts(CharacterTokenizer(), list(map(ord, '<answer>yes</answer>')))['thinking_span_found']
    with pytest.raises(ValueError): response_token_counts(CharacterTokenizer(), [1], 2)
