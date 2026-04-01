"""Headless transport diagnostics — no human listener needed.

Instruments the Daily transport's internal audio write path to measure
timing at every stage. Captures:
  - handle_audio_frame (frame arrives at transport)
  - SOXR resample timing
  - Audio queue depth
  - write_audio_frame duration (how long Daily's C++ SDK blocks)

Usage:
    uv run python diagnostics.py
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

from mistralai.client.models.speechstreamaudiodelta import SpeechStreamAudioDelta
from mistralai.client.sdk import Mistral
from pipecat.frames.frames import (
    EndFrame, TTSAudioRawFrame, TTSSpeakFrame, TTSStartedFrame, TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.transports.daily.transport import DailyParams, DailyTransport

from mistral_tts import MistralTTSService, VOXTRAL_SAMPLE_RATE, _float32le_to_int16

TRANSPORT_RATE = 48000
CHUNK_20MS = int(VOXTRAL_SAMPLE_RATE * 0.02) * 2


class TransportProbe:
    def __init__(self):
        self.events = []
        self._t0 = 0.0
        self.active = False

    def start(self):
        self._t0 = time.monotonic()
        self.events = []
        self.active = True

    def stop(self):
        self.active = False

    def record(self, stage: str, **kwargs):
        if self.active:
            self.events.append((stage, time.monotonic() - self._t0, kwargs))


def patch_transport_deep(transport: DailyTransport, probe: TransportProbe):
    output_transport = transport.output()
    client = output_transport._client
    sender = None
    for _, s in output_transport._media_senders.items():
        sender = s
        break
    if not sender:
        logger.error("[Probe] No media sender found!")
        return

    orig_handle = sender.handle_audio_frame
    orig_resample = sender._resampler.resample
    orig_write = client.write_audio_frame

    async def patched_resample(audio, in_rate, out_rate):
        t0 = time.monotonic()
        result = await orig_resample(audio, in_rate, out_rate)
        probe.record("resample", in_rate=in_rate, out_rate=out_rate,
                      in_bytes=len(audio), out_bytes=len(result),
                      ms=(time.monotonic() - t0) * 1000)
        return result

    async def patched_handle(frame):
        probe.record("handle_entry", size=len(frame.audio), sr=frame.sample_rate,
                      queue_depth=sender._audio_queue.qsize())
        await orig_handle(frame)
        probe.record("handle_exit", queue_depth=sender._audio_queue.qsize())

    async def patched_write(frame):
        t0 = time.monotonic()
        probe.record("write_start", size=len(frame.audio))
        result = await orig_write(frame)
        probe.record("write_done", write_ms=(time.monotonic() - t0) * 1000)
        return result

    sender._resampler.resample = patched_resample
    sender.handle_audio_frame = patched_handle
    client.write_audio_frame = patched_write
    logger.info("[Probe] Instrumented transport output path")


def analyze_probe(probe: TransportProbe, label: str):
    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"{'=' * 72}")
    if not probe.events:
        print("  No events recorded!")
        return

    write_starts = [(t, kw) for stage, t, kw in probe.events if stage == "write_start"]
    write_dones = [(t, kw) for stage, t, kw in probe.events if stage == "write_done"]
    handle_entries = [(t, kw) for stage, t, kw in probe.events if stage == "handle_entry"]
    resamples = [(t, kw) for stage, t, kw in probe.events if stage == "resample"]

    print(f"\n  handle_audio_frame calls: {len(handle_entries)}")
    print(f"  write_audio_frame calls:  {len(write_starts)}")

    if len(write_dones) > 1:
        arr = np.array([kw.get("write_ms", 0) for _, kw in write_dones])
        print(f"\n  write_audio_frame duration:")
        print(f"    min={arr.min():.1f}ms  max={arr.max():.1f}ms  "
              f"mean={arr.mean():.1f}ms  p95={np.percentile(arr, 95):.1f}ms")

    if len(write_starts) > 1:
        gaps = [(write_starts[i][0] - write_starts[i-1][0]) * 1000
                for i in range(1, len(write_starts))]
        arr = np.array(gaps)
        print(f"\n  Gap between write calls:")
        print(f"    min={arr.min():.1f}ms  max={arr.max():.1f}ms  "
              f"mean={arr.mean():.1f}ms  p95={np.percentile(arr, 95):.1f}ms")
        print(f"    Gaps >50ms: {sum(1 for g in gaps if g > 50)}")

    if handle_entries:
        depths = np.array([kw.get("queue_depth", 0) for _, kw in handle_entries])
        print(f"\n  Queue depth at frame arrival:")
        print(f"    min={depths.min()}  max={depths.max()}  mean={depths.mean():.1f}")

    if resamples:
        arr = np.array([kw.get("ms", 0) for _, kw in resamples])
        print(f"\n  Resample: {len(resamples)} calls, "
              f"min={arr.min():.2f}ms  max={arr.max():.2f}ms  total={arr.sum():.1f}ms")


async def prefetch_audio() -> bytes:
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", "").strip())
    voice_id = os.getenv("MISTRAL_VOICE_ID", "gb_jane_neutral")
    stream = await client.audio.speech.complete_async(
        input="Hello! This is a test of the Mistral Voxtral text to speech engine.",
        model="voxtral-mini-tts-2603", voice_id=voice_id,
        response_format="pcm", stream=True,
    )
    raw = bytearray()
    async with stream:
        async for event in stream:
            data = event.data if hasattr(event, "data") else event
            if isinstance(data, SpeechStreamAudioDelta):
                raw.extend(base64.b64decode(data.audio_data))
    return _float32le_to_int16(bytes(raw))


async def create_room() -> str:
    api_key = os.getenv("DAILY_API_KEY", "").strip()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.daily.co/v1/rooms",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"properties": {"exp": int(time.time()) + 3600}},
        ) as resp:
            return (await resp.json())["url"]


async def main():
    prebaked = await prefetch_audio()
    audio_dur = len(prebaked) / (VOXTRAL_SAMPLE_RATE * 2)
    logger.info(f"Prefetched {audio_dur:.1f}s audio")

    room_url = await create_room()
    logger.info(f"Room: {room_url} (no need to join)")

    transport = DailyTransport(
        room_url=room_url, token=None, bot_name="Diagnostics",
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
    task = PipelineTask(pipeline, params=PipelineParams(audio_out_sample_rate=TRANSPORT_RATE))

    probe_a = TransportProbe()
    probe_b = TransportProbe()

    async def run_tests():
        await asyncio.sleep(3)
        patch_transport_deep(transport, probe_a)
        probe_a.start()

        logger.info("=== TEST A: Pre-buffered ===")
        await task.queue_frame(TTSStartedFrame(context_id="a"))
        offset = 0
        while offset < len(prebaked):
            chunk = prebaked[offset:offset + CHUNK_20MS]
            await task.queue_frame(TTSAudioRawFrame(chunk, VOXTRAL_SAMPLE_RATE, 1, context_id="a"))
            offset += CHUNK_20MS
        await task.queue_frame(TTSStoppedFrame(context_id="a"))
        await asyncio.sleep(audio_dur + 3)
        probe_a.stop()

        patch_transport_deep(transport, probe_b)
        probe_b.start()
        logger.info("=== TEST B: Live-streamed ===")
        await task.queue_frame(TTSSpeakFrame(
            text="Hello! This is a test of the Mistral Voxtral text to speech engine."))
        await asyncio.sleep(audio_dur + 6)
        probe_b.stop()

        analyze_probe(probe_a, "TEST A: Pre-buffered (control)")
        analyze_probe(probe_b, "TEST B: Live-streamed (MistralTTSService)")

        await asyncio.sleep(1)
        await task.queue_frame(EndFrame())

    asyncio.create_task(run_tests())
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
