"""Regression test for #6431 (assistant-role image_url chunks crash Bedrock Converse
and are rejected by some Azure models on history replay).

Measured across 14 models: assistant-role image blocks are a per-model property, not
a per-provider one — no Chat-Completions-shaped provider documents support for them,
and Bedrock Converse rejects them by contract (plus an unfixed litellm crash on that
path). The fix strips image_url chunks from assistant messages unconditionally
(previously gated to Anthropic-only), and rewrites the sibling text note — which used
to claim "this image is ALREADY EMBEDDED... do NOT call any file reading tool" — since
that claim becomes false the instant the image_url chunk is removed.

Loaded by source so the suite runs without the pylon runtime (see test_6246's docstring
for the same pattern in this directory): run with
`python3 -m pytest test_6431_assistant_image_strip.py -v` from this directory.
"""

import pathlib


def _load_image_helpers():
    source = (
        pathlib.Path(__file__).resolve().parents[1] / 'utils' / 'image_helpers.py'
    ).read_text()
    source = source.replace('from pylon.core.tools import log', '')

    class _StubLog:
        def error(self, *args, **kwargs):
            pass

    namespace = {'log': _StubLog()}
    exec(compile(source, '<image_helpers>', 'exec'), namespace)  # pylint: disable=W0122
    return namespace


_NS = _load_image_helpers()
strip_image_chunks_from_assistant_messages = _NS['strip_image_chunks_from_assistant_messages']

_TEXT_WITH_FILEPATH = (
    "Image file: sq.png\n"
    "filepath: /attachments/sq.png\n"
    "\n"
    "NOTE: This image is ALREADY EMBEDDED as base64 in this message.\n"
    "Analyze the image directly from the provided image_url data.\n"
    "Do NOT call any file reading tool to re-read this image."
)

_TEXT_NO_FILEPATH = (
    "Image file: sq.png\n"
    "\n"
    "NOTE: This image is ALREADY EMBEDDED as base64 in this message.\n"
    "Analyze the image directly from the provided image_url data.\n"
    "Do NOT call any file reading tool to re-read this image."
)


def _asst_msg(text):
    return {
        'role': 'assistant',
        'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}},
            {'type': 'text', 'text': text},
        ],
    }


def test_image_chunk_removed_and_note_rewritten_with_filepath():
    messages = [_asst_msg(_TEXT_WITH_FILEPATH)]
    strip_image_chunks_from_assistant_messages(messages)

    content = messages[0]['content']
    assert all(c.get('type') != 'image_url' for c in content)

    text = content[0]['text']
    assert 'ALREADY EMBEDDED' not in text
    assert 'do NOT call any file reading tool'.lower() not in text.lower()
    assert 'filepath: /attachments/sq.png' in text
    assert 'file reading tool' in text  # still points at the fallback route


def test_note_rewritten_without_filepath_offers_no_route():
    messages = [_asst_msg(_TEXT_NO_FILEPATH)]
    strip_image_chunks_from_assistant_messages(messages)

    text = messages[0]['content'][0]['text']
    assert 'ALREADY EMBEDDED' not in text
    assert 'filepath' not in text
    assert 'not visible' in text.lower() or 'not carried' in text.lower()


def test_user_role_image_untouched():
    messages = [{
        'role': 'user',
        'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}],
    }]
    strip_image_chunks_from_assistant_messages(messages)
    assert messages[0]['content'][0]['type'] == 'image_url'


def test_assistant_message_without_image_chunk_unchanged():
    messages = [{'role': 'assistant', 'content': [{'type': 'text', 'text': 'plain text'}]}]
    strip_image_chunks_from_assistant_messages(messages)
    assert messages[0]['content'] == [{'type': 'text', 'text': 'plain text'}]


def test_assistant_message_with_only_image_chunk_leaves_empty_content():
    messages = [{
        'role': 'assistant',
        'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}],
    }]
    strip_image_chunks_from_assistant_messages(messages)
    assert messages[0]['content'] == []


def test_unrelated_note_substring_in_context_prompt_not_truncated():
    # context.prompt (echoed as "Context: {context.prompt}") could itself contain the
    # literal string "NOTE:" — must not be mistaken for the producer's own NOTE line.
    text = (
        "Image file: sq.png\n"
        "filepath: /attachments/sq.png\n"
        "Context: please NOTE: focus on the top-left corner\n"
        "\n"
        "NOTE: This image is ALREADY EMBEDDED as base64 in this message.\n"
        "Analyze the image directly from the provided image_url data.\n"
        "Do NOT call any file reading tool to re-read this image."
    )
    messages = [_asst_msg(text)]
    strip_image_chunks_from_assistant_messages(messages)

    result = messages[0]['content'][0]['text']
    assert 'please NOTE: focus on the top-left corner' in result
    assert 'ALREADY EMBEDDED' not in result
    assert 'filepath: /attachments/sq.png' in result


def test_langchain_object_message_handled():
    class FakeAIMessage:
        def __init__(self, content):
            self.type = 'ai'
            self.content = content

    msg = FakeAIMessage([
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}},
        {'type': 'text', 'text': _TEXT_WITH_FILEPATH},
    ])
    strip_image_chunks_from_assistant_messages([msg])

    assert all(c.get('type') != 'image_url' for c in msg.content)
    assert 'ALREADY EMBEDDED' not in msg.content[0]['text']
