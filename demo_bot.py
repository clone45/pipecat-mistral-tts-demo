"""A/B comparison bot: pre-buffered vs live-streamed Mistral TTS.

Runs two tests through the same Daily WebRTC transport:
  Test A: Pre-buffered audio (fetched ahead of time, pushed as 20ms chunks) — SOUNDS PERFECT
  Test B: Live-streamed through MistralTTSService — SOUNDS CHOPPY

Join the printed Daily room URL to listen.

Usage:
    uv run python demo_bot.py
"""

import asyncio
import base64
import os
import sys
import time

import aiohttp
import numpy as np
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

from pipecat.frames.frames import (
    TTSSpeakFrame, TTSAudioRawFrame, EndFrame, TTSStartedFrame, TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.transports.daily.transport import DailyParams, DailyTransport

from mistralai.client.sdk import Mistral
from mistralai.client.models.speechstreamaudiodelta import SpeechStreamAudioDelta

from mistral_tts import MistralTTSService, VOXTRAL_SAMPLE_RATE, _float32le_to_int16

TRANSPORT_RATE = 48000
CHUNK_20MS = int(VOXTRAL_SAMPLE_RATE * 0.02) * 2  # 960 bytes

TEST_SENTENCE = (
    "Hello! This is a test of the Mistral Voxtral text to speech engine "
    "over Daily WebRTC transport. How does this sound to you?"
)


async def prefetch_audio(text: str) -> bytes:
    """Fetch full audio from Mistral API, return as int16 PCM at 24kHz."""
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", "").strip())
    voice_id = os.getenv("MISTRAL_VOICE_ID", "gb_jane_neutral")

    logger.info(f"[Prefetch] Fetching audio...")
    stream = await client.audio.speech.complete_async(
        input=text, model="voxtral-mini-tts-2603",
        voice_id=voice_id, response_format="pcm", stream=True,
    )
    raw = bytearray()
    async with stream:
        async for event in stream:
            data = event.data if hasattr(event, "data") else event
            if isinstance(data, SpeechStreamAudioDelta):
                raw.extend(base64.b64decode(data.audio_data))

    audio = _float32le_to_int16(bytes(raw))
    logger.info(f"[Prefetch] {len(audio) / (VOXTRAL_SAMPLE_RATE * 2):.1f}s ready")
    return audio


async def create_room() -> str:
    api_key = os.getenv("DAILY_API_KEY", "").strip()
    if not api_key:
        print("ERROR: DAILY_API_KEY not set. Copy .env.example to .env and fill in your keys.")
        sys.exit(1)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.daily.co/v1/rooms",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"properties": {"exp": int(time.time()) + 3600}},
        ) as resp:
            if resp.status != 200:
                print(f"ERROR: Failed to create Daily room: {await resp.text()}")
                sys.exit(1)
            data = await resp.json()
            return data["url"]


async def main():
    if not os.getenv("MISTRAL_API_KEY", "").strip():
        print("ERROR: MISTRAL_API_KEY not set. Copy .env.example to .env and fill in your keys.")
        sys.exit(1)

    prebaked = await prefetch_audio(TEST_SENTENCE)
    audio_dur = len(prebaked) / (VOXTRAL_SAMPLE_RATE * 2)

    room_url = await create_room()

    print(f"\n{'=' * 60}")
    print(f"  JOIN THIS ROOM: {room_url}")
    print(f"{'=' * 60}")
    print(f"\n  You will hear 2 tests (~{audio_dur:.0f}s each):")
    print(f"  Test A: Pre-buffered (should sound PERFECT)")
    print(f"  Test B: Live-streamed (currently CHOPPY)")
    print(f"  (3 second gap between tests)\n")

    transport = DailyTransport(
        room_url=room_url, token=None, bot_name="Mistral TTS Demo",
        params=DailyParams(
            audio_in_enabled=True, audio_out_enabled=True,
            audio_out_sample_rate=TRANSPORT_RATE,
        ),
    )

    tts = MistralTTSService(
        api_key=os.getenv("MISTRAL_API_KEY", "").strip(),
        voice_id=os.getenv("MISTRAL_VOICE_ID", "gb_jane_neutral"),
    )

    pipeline = Pipeline([transport.input(), tts, transport.output()])
    runner = PipelineRunner()
    task = PipelineTask(
        pipeline,
        params=PipelineParams(audio_out_sample_rate=TRANSPORT_RATE),
    )

    @transport.event_handler("on_participant_joined")
    async def on_joined(transport, participant):
        if not participant.get("info", {}).get("isLocal"):
            logger.info(f"Participant joined — starting tests")
            await asyncio.sleep(1.5)

            # Test A: Pre-buffered (known good)
            logger.info("=== TEST A: Pre-buffered, 20ms chunks ===")
            ctx = "test_a"
            await task.queue_frame(TTSStartedFrame(context_id=ctx))
            offset = 0
            while offset < len(prebaked):
                chunk = prebaked[offset:offset + CHUNK_20MS]
                await task.queue_frame(
                    TTSAudioRawFrame(chunk, VOXTRAL_SAMPLE_RATE, 1, context_id=ctx)
                )
                offset += CHUNK_20MS
            await task.queue_frame(TTSStoppedFrame(context_id=ctx))
            await asyncio.sleep(audio_dur + 3)

            # Test B: Live-streamed through MistralTTSService
            logger.info("=== TEST B: Live-streamed from Mistral API ===")
            await task.queue_frame(TTSSpeakFrame(text=TEST_SENTENCE))
            await asyncio.sleep(audio_dur + 5)

            await task.queue_frame(EndFrame())

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
