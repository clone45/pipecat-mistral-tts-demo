"""Demo bot: Mistral Voxtral TTS through Pipecat + Daily WebRTC.

Creates a Daily room, joins as a bot, and speaks when a participant joins.

Usage:
    uv run python demo_bot.py

Then open the printed Daily room URL in your browser.
"""

import asyncio
import os
import sys
import time

import aiohttp
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

from pipecat.frames.frames import TTSSpeakFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.transports.daily.transport import DailyParams, DailyTransport

from mistral_tts import MistralTTSService

# Important: transport runs at 48kHz. Pipecat resamples the 24kHz TTS output.
TRANSPORT_RATE = 48000


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
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        print("ERROR: MISTRAL_API_KEY not set. Copy .env.example to .env and fill in your keys.")
        sys.exit(1)

    room_url = await create_room()

    print(f"\n{'=' * 60}")
    print(f"  JOIN THIS ROOM: {room_url}")
    print(f"{'=' * 60}\n")

    transport = DailyTransport(
        room_url=room_url, token=None, bot_name="Mistral TTS Demo",
        params=DailyParams(
            audio_in_enabled=True, audio_out_enabled=True,
            audio_out_sample_rate=TRANSPORT_RATE,
        ),
    )

    tts = MistralTTSService(
        api_key=api_key,
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
            logger.info("Participant joined -- speaking")
            await asyncio.sleep(1)
            await task.queue_frame(TTSSpeakFrame(
                text="Hello! This is the Mistral Voxtral text to speech engine "
                     "running through Pipecat and Daily WebRTC. The audio is "
                     "streamed in real time from Mistral's API. How does it sound?"
            ))
            await asyncio.sleep(15)
            await task.queue_frame(EndFrame())

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
