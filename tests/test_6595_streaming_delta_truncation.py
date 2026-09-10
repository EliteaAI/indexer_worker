"""Regression test for #6595 (live streaming drops a repeated delta token).

`EliteACallback.on_llm_new_token` tried to guess, independently on every chunk,
whether an LLM was streaming cumulative text (the whole string-so-far, repeated
each chunk) or true deltas (only the new token) - using the heuristic "if the
new chunk starts with what we last stored, it's a cumulative resend, so slice
off the overlap." That heuristic silently dropped a genuine delta chunk whenever
it happened to equal the previous chunk: streaming the answer "RRS" as the
delta tokens "R", "R", "S" reached the UI as "RS", because the second "R" was
mistaken for a same-length cumulative resend with no new text.

A first fix locked the cumulative-vs-delta decision from the second chunk alone.
Review on the fix PR pointed out that one chunk is not enough evidence either:
a genuine delta sequence like "a", "ab", "c" looks identical - for that one
comparison - to cumulative growth from "a" to "ab", so locking "cumulative"
there loses the "a"; conversely a genuine cumulative heartbeat like "Hel",
"Hel", "Hello" looks identical - for that one comparison - to a delta token
repeating itself, so locking "delta" there duplicates "Hel".

The current fix (`EliteACallback._compute_stream_delta`) buffers ambiguous
chunks instead of committing to a mode from a single comparison: it waits for
either a chunk that diverges from the buffered chain (proof of delta) or two
consecutive chunks that each strictly extend the previous one (proof of
cumulative) before locking a mode. A run that ends before ever resolving is
flushed by `EliteACallback._flush_stream_delta` (wired into `on_llm_end`) so a
short reply that never disambiguates isn't left stuck in the buffer.

Loaded by source so the suite runs without the pylon runtime.
"""

import ast
import pathlib


def _source(name):
    return (pathlib.Path(__file__).resolve().parents[1] / 'methods' / name).read_text()


def _function(source, name):
    """Return the source text of one top-level or nested function by name."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"function {name} not found")


AGENT_COMMON_SOURCE = _source('agent_common.py')


def _load_functions():
    namespace = {}
    source = (
        "from typing import Dict\n"
        + _function(AGENT_COMMON_SOURCE, '_compute_stream_delta')
        + "\n\n"
        + _function(AGENT_COMMON_SOURCE, '_flush_stream_delta')
    )
    exec(compile(source, '<compute_stream_delta>', 'exec'), namespace)  # pylint: disable=W0122
    return namespace['_compute_stream_delta'], namespace['_flush_stream_delta']


_compute_stream_delta, _flush_stream_delta = _load_functions()


def _stream(chunks):
    """Feed a sequence of raw chunks through the delta computation for one run,
    then flush - as `on_llm_end` does - to surface anything left buffered."""
    last_value_map = {}
    mode_map = {}
    out = []
    for chunk in chunks:
        delta = _compute_stream_delta('run', chunk, last_value_map, mode_map)
        if delta:
            out.append(delta)
    flushed = _flush_stream_delta('run', last_value_map, mode_map)
    if flushed:
        out.append(flushed)
    return ''.join(out)


def test_repeated_delta_token_is_not_dropped():
    # The exact regression: "RRS" streamed as delta tokens "R", "R", "S"
    assert _stream(['R', 'R', 'S']) == 'RRS'


def test_repeated_delta_token_at_stream_start():
    assert _stream(['a', 'a', 'a', 'b']) == 'aaab'


def test_plain_delta_stream_still_works():
    assert _stream(['Hel', 'lo ', 'world']) == 'Hello world'


def test_cumulative_provider_still_works():
    # Some providers resend the full string-so-far on every chunk instead of deltas
    assert _stream(['R', 'RR', 'RRS']) == 'RRS'


def test_cumulative_provider_heartbeat_resend_is_deduped():
    # Once cumulative mode is locked in, an unchanged resend must not duplicate text
    assert _stream(['Hel', 'Hello', 'Hello', 'Hello world']) == 'Hello world'


def test_single_ambiguous_growth_step_does_not_lock_cumulative():
    # Review counter-example: "a" -> "ab" alone is not proof of cumulative growth
    # (it's equally consistent with two independent delta tokens "a" and "ab").
    # A single growth step followed by divergence must resolve as delta, so no
    # text is lost.
    assert _stream(['a', 'ab', 'c']) == 'aabc'


def test_single_ambiguous_repeat_does_not_lock_delta():
    # Review counter-example: "Hel" -> "Hel" alone is not proof of a repeating
    # delta token (it's equally consistent with a cumulative heartbeat that
    # hasn't grown yet). Growth on the next chunk must resolve as cumulative,
    # so no text is duplicated.
    assert _stream(['Hel', 'Hel', 'Hello']) == 'Hello'


def test_unambiguous_divergence_resolves_without_buffering():
    # "b" doesn't extend "a" at all, which a real cumulative resend never
    # does - that's conclusive on its own, no second chunk needed.
    assert _stream(['a', 'b']) == 'ab'


def test_short_repeat_reply_that_never_disambiguates_is_flushed():
    # A two-chunk reply that ends on an unchanged resend, before anything
    # ever proves growth or divergence, must not lose its second chunk to
    # the buffer.
    assert _stream(['a', 'a']) == 'aa'


def test_single_growth_step_that_never_disambiguates_flushes_as_delta():
    # A single growth step ("Hel" -> "Hello") is not proof of cumulative
    # streaming on its own (see test_single_ambiguous_growth_step_does_not_lock_cumulative);
    # if the run ends before a second growth step confirms it, flushing must
    # keep the same delta-biased default as everywhere else in this module
    # rather than guessing cumulative and dropping the anchor's text.
    assert _stream(['Hel', 'Hello']) == 'HelHello'
