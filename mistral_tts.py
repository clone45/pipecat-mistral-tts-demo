"""Mistral Voxtral TTS service for Pipecat.

Streams audio from Mistral's Voxtral TTS API via SSE, converting float32 PCM
to int16 PCM for Pipecat's audio pipeline.

KNOWN ISSUE: Audio is choppy when played through Daily WebRTC transport.
Pre-buffered audio through the same transport sounds perfect.
See README.md for full investigation details.
"""

import asyncio
import base64
import time
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger
from mistralai.client.sdk import Mistral
from mistralai.client.models.speechstreamaudiodelta import SpeechStreamAudioDelta

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

VOXTRAL_SAMPLE_RATE = 24000
YIELD_CHUNK_BYTES = 4800     # 100ms of int16 mono at 24kHz
JITTER_BUFFER_BYTES = 28800  # 600ms jitter buffer


class MistralTTSService(TTSService):
    """Mistral Voxtral TTS service — choppy audio under investigation."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        voice_id: str = "gb_jane_neutral",
        model: str = "voxtral-mini-tts-2603",
        **kwargs,
    ):
        default_settings = TTSSettings(model=model, voice=voice_id, language=None)
        super().__init__(
            sample_rate=VOXTRAL_SAMPLE_RATE,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )
        self._client = Mistral(api_key=api_key)
        self._voice_id = voice_id
        self._model = model

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

        t0 = time.monotonic()
        audio_queue: asyncio.Queue = asyncio.Queue()

        async def fetch_sse():
            try:
                stream = await self._client.audio.speech.complete_async(
                    input=text,
                    model=self._model,
                    voice_id=self._voice_id,
                    response_format="pcm",
                    stream=True,
                )
                chunk_num = 0
                async with stream:
                    async for event in stream:
                        data = event.data if hasattr(event, "data") else event
                        if isinstance(data, SpeechStreamAudioDelta):
                            raw_bytes = base64.b64decode(data.audio_data)
                            pcm_int16 = _float32le_to_int16(raw_bytes)
                            if len(pcm_int16) > 0:
                                chunk_num += 1
                                await audio_queue.put(pcm_int16)
                await audio_queue.put(None)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[fetch_sse] Error: {e}")
                await audio_queue.put(e)

        fetch_task = asyncio.create_task(fetch_sse())

        try:
            await self.start_tts_usage_metrics(text)
        except Exception:
            pass

        buffer = bytearray()
        chunk_count = 0
        total_audio_bytes = 0
        ttfb = None
        started_yielding = False

        try:
            while True:
                if not started_yielding and len(buffer) < JITTER_BUFFER_BYTES:
                    item = await audio_queue.get()
                    if item is None:
                        started_yielding = True
                    elif isinstance(item, Exception):
                        yield ErrorFrame(error=f"Mistral TTS error: {item}")
                        break
                    else:
                        buffer.extend(item)
                        if ttfb is None:
                            ttfb = time.monotonic() - t0
                            try:
                                await self.stop_ttfb_metrics()
                            except Exception:
                                pass
                        continue

                started_yielding = True

                if len(buffer) < YIELD_CHUNK_BYTES:
                    item = await audio_queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        yield ErrorFrame(error=f"Mistral TTS error: {item}")
                        break
                    buffer.extend(item)

                while len(buffer) >= YIELD_CHUNK_BYTES:
                    chunk = bytes(buffer[:YIELD_CHUNK_BYTES])
                    del buffer[:YIELD_CHUNK_BYTES]
                    chunk_count += 1
                    total_audio_bytes += len(chunk)
                    yield TTSAudioRawFrame(
                        chunk, self.sample_rate, 1, context_id=context_id,
                    )
                    await asyncio.sleep(0.01)

            if len(buffer) > 0:
                aligned_len = len(buffer) & ~1
                if aligned_len > 0:
                    chunk_count += 1
                    total_audio_bytes += aligned_len
                    yield TTSAudioRawFrame(
                        bytes(buffer[:aligned_len]),
                        self.sample_rate, 1, context_id=context_id,
                    )
        finally:
            fetch_task.cancel()
            try:
                await fetch_task
            except asyncio.CancelledError:
                pass

        elapsed = time.monotonic() - t0
        audio_duration = total_audio_bytes / (self.sample_rate * 2) if self.sample_rate else 0
        ttfb_ms = ttfb * 1000 if ttfb else 0
        logger.info(
            f"[Mistral TTS] \"{text[:50]}{'...' if len(text) > 50 else ''}\" "
            f"ttfb={ttfb_ms:.0f}ms total={elapsed * 1000:.0f}ms "
            f"chunks={chunk_count} audio={audio_duration:.1f}s"
        )


def _float32le_to_int16(data: bytes) -> bytes:
    samples = np.frombuffer(data, dtype=np.float32)
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()
