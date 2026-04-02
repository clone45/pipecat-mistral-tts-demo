"""Mistral Voxtral TTS service for Pipecat.

Streams audio from Mistral's Voxtral TTS API via SSE using raw aiohttp
HTTP streaming. This bypasses the Mistral SDK's event parser, which batches
audio into large chunks that cause choppy playback through WebRTC transports.

Reading directly from the HTTP response with iter_any() lets the network I/O
naturally pace the event loop, matching the pattern used by Pipecat's built-in
NeuphonicHttpTTSService.

Requires:
    pip install pipecat-ai[daily] aiohttp numpy

Example:
    tts = MistralTTSService(
        api_key="your-api-key",
        voice_id="gb_jane_neutral",
    )
"""

import aiohttp
import asyncio
import base64
import json
import time
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

# Voxtral always outputs 24kHz audio
VOXTRAL_SAMPLE_RATE = 24000


def _float32le_to_int16(data: bytes) -> bytes:
    """Convert raw float32 little-endian PCM to int16 PCM.

    Voxtral's PCM streaming format outputs float32 LE samples in [-1.0, 1.0].
    Pipecat's audio pipeline expects int16 PCM (2 bytes per sample).
    """
    samples = np.frombuffer(data, dtype=np.float32)
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


class MistralTTSService(TTSService):
    """Mistral Voxtral TTS service using raw HTTP streaming.

    Streams audio via SSE from Mistral's /v1/audio/speech endpoint using
    aiohttp's raw byte reader. Voxtral outputs float32 LE PCM at 24kHz;
    this service converts to int16 PCM for Pipecat's audio pipeline.

    Important: The transport's audio_out_sample_rate should be set to 48000
    (not 24000). Pipecat's SOXR resampler handles the 24kHz -> 48kHz
    conversion automatically. Running the transport at 24kHz causes choppy
    playback due to Daily's WebRTC audio engine expecting 48kHz.

    Voice IDs can be preset slugs (e.g. "gb_jane_neutral", "en_paul_cheerful")
    or custom voice UUIDs created via the Mistral Voices API.

    Args:
        api_key: Mistral API key.
        voice_id: Preset slug or custom voice UUID. Defaults to "gb_jane_neutral".
        model: Voxtral model ID. Defaults to "voxtral-mini-tts-2603".
        aiohttp_session: Optional shared aiohttp session. If not provided,
            a new session is created per request.
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str = "gb_jane_neutral",
        model: str = "voxtral-mini-tts-2603",
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
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
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

        t0 = time.monotonic()

        url = "https://api.mistral.ai/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload = {
            "model": self._model,
            "input": text,
            "voice_id": self._voice_id,
            "response_format": "pcm",
            "stream": True,
        }

        session = self._session
        should_close = False
        if session is None:
            session = aiohttp.ClientSession()
            should_close = True

        chunk_count = 0
        total_audio_bytes = 0
        ttfb = None

        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"{self}: Mistral API error: {response.status} - {error_text}")
                    yield ErrorFrame(error=f"Mistral API error: {response.status}")
                    return

                try:
                    await self.start_tts_usage_metrics(text)
                except Exception:
                    pass

                # Read the SSE stream using iter_any() which yields bytes as
                # they arrive from the network, then reassemble into lines.
                #
                # Why not use the Mistral SDK or response.content's line iterator?
                # - The Mistral SDK's SSE parser batches audio into large chunks
                #   (up to 96KB), which blocks the event loop and starves the
                #   Daily WebRTC transport's audio write thread.
                # - aiohttp's line iterator has a high_water limit (~64KB) that
                #   Mistral's base64 audio lines exceed, causing "Chunk too big".
                #
                # iter_any() reads raw TCP chunks and naturally yields control
                # to the event loop between reads, letting the transport process
                # audio smoothly. This matches Pipecat's NeuphonicHttpTTSService.
                pending = b""
                async for raw_chunk in response.content.iter_any():
                    pending += raw_chunk
                    while b"\n" in pending:
                        line_bytes, pending = pending.split(b"\n", 1)
                        message = line_bytes.decode("utf-8", errors="ignore").strip()

                        if not message or not message.startswith("data:"):
                            continue

                        data_content = message[len("data:"):].strip()
                        if data_content == "[DONE]":
                            break

                        try:
                            parsed = json.loads(data_content)
                            audio_b64 = parsed.get("audio_data")

                            if audio_b64:
                                raw_bytes = base64.b64decode(audio_b64)
                                pcm_int16 = _float32le_to_int16(raw_bytes)

                                if len(pcm_int16) > 0:
                                    if ttfb is None:
                                        ttfb = time.monotonic() - t0
                                    try:
                                        await self.stop_ttfb_metrics()
                                    except Exception:
                                        pass

                                    chunk_count += 1
                                    total_audio_bytes += len(pcm_int16)

                                    yield TTSAudioRawFrame(
                                        audio=pcm_int16,
                                        sample_rate=self.sample_rate,
                                        num_channels=1,
                                        context_id=context_id,
                                    )

                        except json.JSONDecodeError:
                            logger.warning(f"{self}: Failed to parse SSE JSON: {data_content[:100]}")
                        except Exception as e:
                            logger.error(f"{self}: Error processing audio chunk: {e}")

        except asyncio.CancelledError:
            logger.debug(f"{self}: TTS generation cancelled")
            raise
        except Exception as e:
            logger.error(f"{self}: Mistral TTS error: {e}")
            yield ErrorFrame(error=f"Mistral TTS error: {e}")
        finally:
            if should_close:
                await session.close()
            try:
                await self.stop_ttfb_metrics()
            except Exception:
                pass

        elapsed = time.monotonic() - t0
        audio_duration = total_audio_bytes / (self.sample_rate * 2) if self.sample_rate else 0
        ttfb_ms = ttfb * 1000 if ttfb else 0
        logger.info(
            f"[Mistral TTS] \"{text[:50]}{'...' if len(text) > 50 else ''}\" "
            f"ttfb={ttfb_ms:.0f}ms total={elapsed * 1000:.0f}ms "
            f"chunks={chunk_count} audio={audio_duration:.1f}s"
        )
