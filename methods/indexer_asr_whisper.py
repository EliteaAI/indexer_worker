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

""" Method: Whisper transcription via indexer worker (task_node_light / thread pool) """

import io
import wave

from pylon.core.tools import log, web
from tools import worker_core

# Internal event-node channel names (indexer → pylon_main)
_EN_ASR_TRANSCRIPT_DONE = "voice_asr_transcript_done"
_EN_ASR_ERROR = "voice_asr_error"


class Method:

    @web.method()
    def indexer_asr_whisper(
        self,
        *args,
        sid: str = "",
        project_id: int = 0,
        project_llm_key: str = "",
        model_name: str = "",
        language: str = "en",
        audio_bytes: bytes = b"",
        **kwargs,
    ):
        """
        Transcribe a PCM16 audio buffer via the LiteLLM proxy and emit the
        transcript to pylon_main via the event_node.

        Runs in a task_node_light thread (short-lived, IO-bound).  Called once
        per buffered speech segment by the VAD logic in pylon_main (sio/asr.py).

        Payload convention:
          voice_asr_transcript_done  {"sid": str, "transcript": str}
          voice_asr_error            {"sid": str, "error": str}
        """
        local_event_node = worker_core.event_node

        if not audio_bytes:
            return

        try:
            transcript = _call_whisper(project_id, project_llm_key, model_name, language, audio_bytes)
            if transcript:
                local_event_node.emit(_EN_ASR_TRANSCRIPT_DONE, {"sid": sid, "transcript": transcript})
        except Exception as exc:
            import requests as _req
            if isinstance(exc, _req.HTTPError) and exc.response is not None and exc.response.status_code == 429:
                # Rate-limited — drop this chunk silently so recording stays alive
                log.debug("indexer_asr_whisper: rate-limited (429) for sid=%s, dropping chunk", sid)
            else:
                log.error("indexer_asr_whisper: transcription error for sid=%s: %s", sid, exc)
                local_event_node.emit(_EN_ASR_ERROR, {"sid": sid, "error": str(exc)})


def _pcm16_to_wav(pcm_data: bytes, sample_rate: int = 24000) -> io.BytesIO:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    buf.seek(0)
    return buf


_LITELLM_ASR_URL = "http://127.0.0.1:8081/v1/audio/transcriptions"


def _call_whisper(
    project_id: int,
    project_llm_key: str,
    model_name: str,
    language: str,
    pcm_data: bytes,
) -> str:
    import requests

    wav_buf = _pcm16_to_wav(pcm_data)
    litellm_model = f"{project_id}_{model_name}"

    response = requests.post(
        _LITELLM_ASR_URL,
        headers={"Authorization": f"Bearer {project_llm_key}"},
        files={"file": ("audio.wav", wav_buf, "audio/wav")},
        data={"model": litellm_model, "language": language},
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("text", "")
