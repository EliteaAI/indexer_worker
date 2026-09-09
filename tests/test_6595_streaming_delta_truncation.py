"""Regression test for #6595 (live streaming drops a repeated delta token).

`EliteACallback.on_llm_new_token` tried to guess, independently on every chunk,
whether an LLM was streaming cumulative text (the whole string-so-far, repeated
each chunk) or true deltas (only the new token) - using the heuristic "if the
new chunk starts with what we last stored, it's a cumulative resend, so slice
off the overlap." That heuristic silently dropped a genuine delta chunk whenever
it happened to equal the previous chunk: streaming the answer "RRS" as the
delta tokens "R", "R", "S" reached the UI as "RS", because the second "R" was
mistaken for a same-length cumulative resend with no new text.

The fix (`EliteACallback._compute_stream_delta`) locks in the cumulative-vs-delta
decision once per run - using the second chunk as the only ambiguous data point,
where a *strictly longer* prefix match is the sole evidence accepted as proof of
cumulative streaming - instead of re-guessing on every chunk.

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


def _load_compute_stream_delta():
    func_source = _function(AGENT_COMMON_SOURCE, '_compute_stream_delta')
    namespace = {}
    exec(compile("from typing import Dict\n" + func_source, '<compute_stream_delta>', 'exec'),
         namespace)  # pylint: disable=W0122
    return namespace['_compute_stream_delta']


_compute_stream_delta = _load_compute_stream_delta()


def _stream(chunks):
    """Feed a sequence of raw chunks through the delta computation for one run."""
    last_value_map = {}
    mode_map = {}
    out = []
    for chunk in chunks:
        delta = _compute_stream_delta('run', chunk, last_value_map, mode_map)
        if delta:
            out.append(delta)
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
