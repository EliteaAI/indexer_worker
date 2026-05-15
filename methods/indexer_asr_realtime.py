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

"""
Method: Realtime ASR WebSocket relay via indexer worker (task_node_light / thread pool).

Architecture:
  pylon_main                        indexer (this task)
  ──────────────                    ──────────────────
  asr_audio_chunk (SIO)
    → voice_asr_audio_input_{sid}   → ws.send(audio_buffer.append)
  asr_stop (SIO)
    → voice_asr_stop_{sid}          → sets stop_event, ws.close()
                                    ← ws.on_message (transcription events)
                                      → voice_asr_transcript_delta / _done
                                        → pylon_main handler → SIO browser
"""

import json
import threading

from pylon.core.tools import log, web
from tools import worker_core

# Internal event-node channel names (indexer → pylon_main)
_EN_ASR_TRANSCRIPT_DELTA = "voice_asr_transcript_delta"
_EN_ASR_TRANSCRIPT_DONE = "voice_asr_transcript_done"
_EN_ASR_ERROR = "voice_asr_error"

# Internal event-node channel names (pylon_main → indexer)
_EN_ASR_AUDIO_INPUT_PREFIX = "voice_asr_audio_input_"
_EN_ASR_STOP_PREFIX = "voice_asr_stop_"


class Method:

    @web.method()
    def indexer_asr_realtime(
        self,
        *args,
        sid: str = "",
        project_id: int = 0,
        project_llm_key: str = "",
        model_name: str = "",
        language: str = "en",
        **kwargs,
    ):
        """
        Manage a long-lived WebSocket connection to the provider realtime ASR endpoint.

        Runs in a task_node_light thread (long-lived, IO-bound).

        - Subscribes to voice_asr_audio_input_{sid} to receive audio from pylon_main
          and forward it to the provider WebSocket.
        - Subscribes to voice_asr_stop_{sid} to receive a stop signal.
        - Emits voice_asr_transcript_delta / voice_asr_transcript_done back to pylon_main.

        Payload convention (pylon_main → indexer via event_node):
          voice_asr_audio_input_{sid}  {"audio": bytes}  (raw PCM16 bytes)

        Payload convention (indexer → pylon_main via event_node):
          voice_asr_transcript_delta   {"sid": str, "delta": str}
          voice_asr_transcript_done    {"sid": str, "transcript": str}
          voice_asr_error              {"sid": str, "error": str}
        """
        local_event_node = worker_core.event_node

        litellm_model = f"{project_id}_{model_name}"
        ws_url = f"ws://127.0.0.1:8081/v1/realtime?model={litellm_model}&intent=transcription"
        ws_headers = [f"Authorization: Bearer {project_llm_key}", "OpenAI-Beta: realtime=v1"]

        stop_event = threading.Event()
        audio_input_channel = f"{_EN_ASR_AUDIO_INPUT_PREFIX}{sid}"
        stop_channel = f"{_EN_ASR_STOP_PREFIX}{sid}"

        # WebSocket state
        ws_state = {"ws": None, "connected": False, "lock": threading.Lock(), "queue": []}

        def _on_audio_input(event, payload, *a):
            import base64 as _b64
            pcm_bytes = payload.get("audio", b"")
            if not pcm_bytes:
                return
            audio_b64 = _b64.b64encode(pcm_bytes).decode("ascii")
            with ws_state["lock"]:
                if not ws_state["connected"]:
                    ws_state["queue"].append(audio_b64)
                    return
                ws = ws_state["ws"]
            if ws:
                try:
                    ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64,
                    }))
                except Exception as exc:
                    log.warning(f"indexer_asr_realtime: send error for sid={sid}: {exc}")

        def _on_stop(event, payload, *a):
            stop_event.set()
            ws = ws_state.get("ws")
            if ws:
                try:
                    ws.close()
                except Exception:
                    pass

        local_event_node.subscribe(audio_input_channel, _on_audio_input)
        local_event_node.subscribe(stop_channel, _on_stop)

        try:
            _run_realtime_ws(
                local_event_node, sid, ws_url, ws_headers or [],
                model_name, language, ws_state, stop_event,
            )
        finally:
            local_event_node.unsubscribe(audio_input_channel, _on_audio_input)
            local_event_node.unsubscribe(stop_channel, _on_stop)


def _run_realtime_ws(
    local_event_node,
    sid: str,
    ws_url: str,
    ws_headers: list,
    model_name: str,
    language: str,
    ws_state: dict,
    stop_event: threading.Event,
) -> None:
    import websocket as ws_lib  # websocket-client

    def _on_open(ws):
        ws.send(json.dumps({
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": model_name or "gpt-4o-transcribe",
                    "language": language,
                },
                "turn_detection": {
                    "type": "server_vad",
                    "silence_duration_ms": 300,
                    "threshold": 0.7,
                },
            },
        }))
        with ws_state["lock"]:
            ws_state["connected"] = True
            for audio in ws_state["queue"]:
                try:
                    ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio}))
                except Exception:
                    pass
            ws_state["queue"] = []

    def _on_message(ws, raw):
        try:
            msg = json.loads(raw)
        except Exception:
            return
        event_type = msg.get("type", "")

        if event_type == "error":
            log.error(f"indexer_asr_realtime: provider error for sid={sid}: {msg.get('error')}")
            return

        if event_type in (
            "conversation.item.input_audio_transcription.delta",
            "response.audio_transcript.delta",
        ):
            delta = msg.get("delta", "")
            if delta:
                local_event_node.emit(_EN_ASR_TRANSCRIPT_DELTA, {"sid": sid, "delta": delta})

        elif event_type in (
            "conversation.item.input_audio_transcription.completed",
            "response.audio_transcript.done",
        ):
            transcript = msg.get("transcript", "")
            local_event_node.emit(_EN_ASR_TRANSCRIPT_DONE, {"sid": sid, "transcript": transcript})

    def _on_error(ws, error):
        if not stop_event.is_set():
            log.error(f"indexer_asr_realtime: WS error for sid={sid}: {error}")
            local_event_node.emit(_EN_ASR_ERROR, {"sid": sid, "error": str(error)})
        stop_event.set()

    def _on_close(ws, code, reason):
        stop_event.set()

    ws = ws_lib.WebSocketApp(
        ws_url,
        header=ws_headers,
        on_open=_on_open,
        on_message=_on_message,
        on_error=_on_error,
        on_close=_on_close,
    )
    ws_state["ws"] = ws

    # run_forever blocks until the WebSocket closes or stop_event triggers a close
    ws.run_forever()
