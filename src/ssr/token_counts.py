"""Count actual response token IDs, rather than retokenizing stripped reasoning."""
import re


def response_token_counts(tokenizer, ids, injected_tokens=0):
    if not 0 <= injected_tokens <= len(ids):
        raise ValueError('Invalid injected-token count')
    decode = lambda n: tokenizer.decode(ids[:n], skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
    raw = decode(len(ids))
    opening = re.search(r'<(reason|thinking|think)>', raw)
    begin = end = 0
    if opening:
        start_char = opening.end()
        closing = re.search(r'</' + opening[1] + r'>|<search>|<text_search>|<tool_call>|<answer>|<\|im_end\|>|<\|endoftext\|>', raw[start_char:])
        end_char = start_char + closing.start() if closing else len(raw)

        def first_true(predicate):
            lo, hi = 0, len(ids)
            while lo < hi:
                mid = (lo + hi) // 2
                if predicate(decode(mid)):
                    hi = mid
                else:
                    lo = mid + 1
            return lo

        if end_char > start_char:
            # Count tokens overlapping the body, including whitespace. Prefix
            # matching at its end handles multi-token Unicode characters.
            begin = max(0, first_true(lambda s: len(s) > start_char) - 1)
            end = first_true(lambda s: s.startswith(raw[:end_char]))
        else:
            begin = end = first_true(lambda s: len(s) >= start_char)
    return {
        'total_generated_tokens': len(ids) - injected_tokens,
        'total_response_tokens': len(ids),
        'thinking_tokens': end - begin,
        'generated_thinking_tokens': max(0, end - max(begin, injected_tokens)),
        'thinking_span_found': opening is not None,
        'thinking_span_token_start': begin if opening else None,
        'thinking_span_token_end': end if opening else None,
    }
