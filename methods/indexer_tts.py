#!/usr/bin/python3
# coding=utf-8

#   Copyright 2024 EPAM Systems
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

""" Method: TTS streaming via indexer worker (task_node_light / thread pool) """

import re
import threading

from pylon.core.tools import log, web
from tools import worker_core

from ..utils.voice_router import register as voice_register, unregister as voice_unregister
from ..utils.voice_router import TTS_CANCEL, TTS_NEXT

# Internal event-node channel names (indexer → pylon_main)
_EN_TTS_AUDIO_CHUNK = "voice_tts_audio_chunk"
_EN_TTS_DONE = "voice_tts_done"
_EN_TTS_ERROR = "voice_tts_error"

# Sentence boundary: period/exclamation/question followed by whitespace, or one+ newlines.
# Splitting here lets the frontend receive exact audio-to-text anchor points (char_end waypoints).
_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+|\n+')


def _split_sentences(text: str) -> list[tuple[str, int]]:
    """
    Split *text* into sentence segments.

    Returns a list of ``(sentence_text, char_end)`` tuples where ``char_end``
    is the character offset in the original *text* string immediately after
    the last character of that sentence (i.e. before any inter-sentence
    whitespace / newline).  The frontend records these offsets as exact
    audio→text waypoints so the word-highlight can interpolate accurately
    between sentence boundaries.
    """
    if not text.strip():
        return [(text, len(text))]

    result = []
    prev = 0
    for m in _SENTENCE_BOUNDARY_RE.finditer(text):
        segment = text[prev:m.start()]
        if segment.strip():
            result.append((segment.strip(), m.start()))
        prev = m.end()

    # Trailing segment (no boundary after it)
    tail = text[prev:]
    if tail.strip():
        result.append((tail.strip(), len(text)))

    return result if result else [(text, len(text))]


class Method:

    @web.method()
    def indexer_tts(
        self,
        *args,
        sid: str = "",
        project_id: int = 0,
        project_llm_key: str = "",
        model_name: str = "",
        text: str = "",
        voice: str = "alloy",
        speed: float = 1.0,
        voice_instructions: str = "",
        **kwargs,
    ):
        """
        Stream TTS audio from the LiteLLM proxy and relay each chunk to
        pylon_main via the event_node.

        All provider routing (OpenAI, Azure OpenAI, ElevenLabs, Vertex AI,
        AWS Polly, …) is handled by the LiteLLM proxy.  The project_llm_key
        is resolved by pylon_main (which has VaultClient access) and passed
        in as a task kwarg.

        The text is split into sentences.  Each sentence is sent as a separate
        TTS request so the frontend receives ``tts_done`` waypoints that carry
        the exact ``char_end`` position in the speakable text after each
        sentence.  The frontend uses these waypoints for linear interpolation
        of the word-highlight position, giving near-exact sync.

        Runs in a task_node_light thread (IO-bound, no fork overhead).

        Event payload convention:
          voice_tts_audio_chunk  {"sid": str, "audio": bytes, "sample_rate": int}
          voice_tts_done         {"sid": str}                         — final
          voice_tts_done         {"sid": str, "char_end": int}        — sentence waypoint
          voice_tts_error        {"sid": str, "error": str}
        """
        local_event_node = worker_core.event_node

        cancel_event = threading.Event()
        next_event = threading.Event()
        next_config = {}

        def _on_cancel(payload):
            cancel_event.set()
            # Unblock any waiting next_event so the loop exits cleanly
            next_event.set()

        def _on_next(payload):
            new_voice = payload.get("voice")
            new_speed = payload.get("speed")
            if new_voice is not None:
                next_config["voice"] = new_voice
            if new_speed is not None:
                next_config["speed"] = float(new_speed)
            next_event.set()

        voice_register(local_event_node, sid, TTS_CANCEL, _on_cancel)
        voice_register(local_event_node, sid, TTS_NEXT, _on_next)
        try:
            _run_tts_stream(
                local_event_node, sid, project_id, project_llm_key,
                model_name, text, voice, speed, cancel_event, voice_instructions,
                next_event=next_event,
                next_config=next_config,
            )
        finally:
            voice_unregister(sid, TTS_CANCEL, TTS_NEXT)


_LITELLM_TTS_URL = "http://127.0.0.1:8081/v1/audio/speech"

_DEFAULT_GPT4O_TTS_INSTRUCTIONS = (
    "Affect: calm and warm. "
    "Pacing: steady and measured, never rushing. "
    "Tone: conversational and clear. "
    "Do not change your speaking style, pitch, or rhythm between sentences."
)


def _build_request_params(project_id: int, project_llm_key: str) -> tuple[str, dict]:
    """Return (url, headers) for the LiteLLM TTS endpoint."""
    url = _LITELLM_TTS_URL
    headers = {
        "Authorization": f"Bearer {project_llm_key}",
        "Content-Type": "application/json",
    }
    return url, headers


def _get_tone_params(model_name: str, sentences: list, idx: int, voice_instructions: str = "") -> dict:
    """
    Return provider-specific extra payload fields that help the TTS model
    maintain a consistent tone across separate per-sentence API calls.

    - ElevenLabs (model name contains "eleven"): passes ``previous_text`` /
      ``next_text`` so the model has prosody context from adjacent sentences.
    - OpenAI gpt-4o-*-tts: passes an ``instructions`` field that pins the
      model to a specific voice persona.  Uses ``voice_instructions`` when
      provided, otherwise falls back to ``_DEFAULT_GPT4O_TTS_INSTRUCTIONS``.
    - All other models: returns an empty dict (no change in behaviour).
    """
    lower = model_name.lower()

    if "eleven" in lower:
        params = {}
        if idx > 0:
            params["previous_text"] = " ".join(s for s, _ in sentences[:idx])
        if idx < len(sentences) - 1:
            params["next_text"] = " ".join(s for s, _ in sentences[idx + 1:])
        return params

    if "gpt-4o" in lower and "tts" in lower:
        instructions = voice_instructions.strip() or _DEFAULT_GPT4O_TTS_INSTRUCTIONS
        return {"instructions": instructions}

    return {}


def _stream_sentence(
    local_event_node,
    sid: str,
    url: str,
    headers: dict,
    model_name: str,
    voice: str,
    speed: float,
    sentence: str,
    cancel_event: threading.Event,
    extra_params: dict = None,
) -> bool:
    """
    Stream one sentence's PCM audio to pylon_main via event_node.
    Returns True on success, False on error or cancellation.

    PCM bytes are accumulated in a local buffer and only emitted once the
    buffer reaches MIN_EMIT_BYTES.  This prevents the HTTP chunked-transfer
    encoder from producing tiny trailing fragments (1–4 bytes) that the
    frontend would schedule as sub-quantum AudioBufferSourceNodes, causing
    audible pops.  The remainder is always flushed at the end of the sentence
    so the sentence boundary stays clean and aligned with the tts_done event.
    """
    import requests as req

    # 8 192 bytes = 4 096 PCM-16 samples at 24 kHz ≈ 170 ms.
    # Large enough that HTTP chunk-boundary fragments are always absorbed;
    # small enough to keep streaming latency imperceptible.
    MIN_EMIT_BYTES = 8192

    payload = {
        "model": model_name,
        "input": sentence,
        "voice": voice,
        "speed": speed,
        "response_format": "pcm",
        **(extra_params or {}),
    }
    try:
        accumulator = bytearray()
        with req.post(url, headers=headers, json=payload, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=4096):
                if cancel_event.is_set():
                    return False
                if chunk:
                    accumulator.extend(chunk)
                    if len(accumulator) >= MIN_EMIT_BYTES:
                        local_event_node.emit(
                            _EN_TTS_AUDIO_CHUNK,
                            {"sid": sid, "audio": bytes(accumulator), "sample_rate": 24000},
                        )
                        accumulator = bytearray()
        # Flush the remainder so all of this sentence's audio arrives before
        # the tts_done waypoint is emitted by the caller.
        if accumulator and not cancel_event.is_set():
            local_event_node.emit(
                _EN_TTS_AUDIO_CHUNK,
                {"sid": sid, "audio": bytes(accumulator), "sample_rate": 24000},
            )
    except Exception as exc:
        if not cancel_event.is_set():
            log.error(f"indexer_tts streaming error for sid={sid}: {exc}")
            local_event_node.emit(_EN_TTS_ERROR, {"sid": sid, "error": str(exc)})
        return False
    return True


def _run_tts_stream(
    local_event_node,
    sid: str,
    project_id: int,
    project_llm_key: str,
    model_name: str,
    text: str,
    voice: str,
    speed: float,
    cancel_event: threading.Event,
    voice_instructions: str = "",
    next_event: threading.Event = None,
    next_config: dict = None,
) -> None:
    url, headers = _build_request_params(project_id, project_llm_key)
    # LiteLLM expects models in "{project_id}_{model_name}" format for project-scoped routing
    litellm_model = f"{project_id}_{model_name}"
    sentences = _split_sentences(text)

    for i, (sentence, char_end) in enumerate(sentences):
        if cancel_event.is_set():
            return

        extra_params = _get_tone_params(model_name, sentences, i, voice_instructions)
        ok = _stream_sentence(
            local_event_node, sid, url, headers,
            litellm_model, voice, speed, sentence, cancel_event,
            extra_params=extra_params,
        )
        if not ok:
            return

        if cancel_event.is_set():
            return

        is_last = (i == len(sentences) - 1)
        if is_last:
            # Final done — no char_end; frontend uses this to set totalDuration
            local_event_node.emit(_EN_TTS_DONE, {"sid": sid})
        else:
            # Sentence waypoint — frontend records exact audio→text anchor, then
            # ACKs with tts_next (carrying latest voice/speed) before we proceed.
            local_event_node.emit(_EN_TTS_DONE, {"sid": sid, "char_end": char_end})
            if next_event is not None:
                next_event.clear()
                next_event.wait(timeout=30)
                if cancel_event.is_set():
                    return
                # Apply any settings update carried in the ACK
                if next_config:
                    voice = next_config.pop("voice", voice)
                    speed = next_config.pop("speed", speed)
