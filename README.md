# Pipecat + Mistral Voxtral TTS: Choppy Audio Issue

## The Problem

When streaming audio from Mistral's Voxtral TTS API through Pipecat's pipeline to a Daily WebRTC transport, the audio is **choppy with audible gaps**. Pre-buffered audio through the exact same pipeline and transport sounds **perfect**.

## Quick Demo

```bash
# 1. Install dependencies
uv sync

# 2. Copy and fill in your API keys
cp .env.example .env
# Edit .env with your Daily and Mistral API keys

# 3. Run the A/B comparison
uv run python demo_bot.py
```

Join the printed Daily room URL. You'll hear:
- **Test A (pre-buffered):** Perfect, smooth audio
- **Test B (live-streamed):** Choppy, with audible gaps

Both tests use the same audio content, same transport, same sample rate. The only difference is how frames enter the pipeline.

## Headless Diagnostics (no listener needed)

```bash
uv run python diagnostics.py
```

This instruments the Daily transport internals and prints timing data showing where the gaps originate.

## What We Know

### Ruled Out
- **Mistral's API delivery speed:** SSE timing analysis proves audio arrives at 4-8x real-time with zero theoretical underruns
- **Audio data quality:** WAV files captured at every pipeline stage sound perfect
- **float32-to-int16 conversion:** Verified correct, takes <0.2ms per chunk
- **Resampling performance:** SOXR takes 0.01-3ms per call, negligible

### The Core Issue

When pre-buffered audio is pushed all at once, the Daily transport's internal audio queue stays deep (~60-130 frames). Daily's `CustomAudioSource.write_frames()` returns at a rock-steady 40ms cadence. **Zero gaps.**

When audio is live-streamed, frames arrive at the transport in bursts (gated by SSE event timing). Between bursts, the queue drains. When the queue is shallow or empty, Daily's write timing becomes erratic (7ms to 311ms instead of steady 40ms). These erratic writes are the audible choppiness.

### Transport-Level Measurements

| Metric | Pre-buffered (perfect) | Live-streamed (choppy) |
|--------|----------------------|----------------------|
| write_audio_frame duration | 39.7-45ms (steady) | 7-311ms (erratic) |
| Gaps >50ms between writes | 0 | 5-17 |
| Queue depth at frame arrival | min=1, avg=62 | min=0, avg=25-60 |

### What We've Tried

1. **Buffering to `self.chunk_size` (0.5s)** before yielding — matches Hume/OpenAI pattern. Doesn't help because subsequent yields are still gated by SSE arrival timing.

2. **Larger initial jitter buffer (0.6-2.0s)** — reduces max gap but doesn't eliminate them. Gaps occur throughout the stream, not just at the start.

3. **Re-chunking to uniform 20ms frames** — the resampler still returns 0 bytes for half the calls (SOXR VHQ internal buffering), so queue fills at half the expected rate.

4. **Decoupled producer/consumer** with `asyncio.Queue` — background task reads SSE, generator yields from buffer. Helps structure the code but doesn't fix the fundamental timing issue.

5. **100ms yield chunks + 600ms jitter buffer + `asyncio.sleep(0.01)` between yields** — still choppy.

6. **48kHz transport rate** (letting Pipecat resample 24kHz TTS to 48kHz) — necessary (24kHz transport is worse) but not sufficient alone.

### Key Technical Details

- Pipecat: 0.0.108
- Mistral SDK: `mistralai` 2.2.0
- Transport: DailyTransport at 48kHz
- TTS output: 24kHz int16 mono (converted from Mistral's float32 LE PCM)
- Mistral streams via SSE (`speech.audio.delta` events), not HTTP chunked transfer
- SOXR stream resampler clears internal state after 200ms without data (`CLEAR_STREAM_AFTER_SECS=0.2` in `pipecat/audio/resamplers/soxr_stream_resampler.py`)
- Pipecat's `BaseOutputTransport` re-chunks all frames to `_audio_chunk_size` before queuing, so input frame size shouldn't matter — yet it does

### Possibly Relevant

- Hume AI's TTS (48kHz, HTTP JSON streaming) works perfectly through the same Pipecat pipeline. Hume uses `synthesize_json_streaming()` which delivers chunks more steadily than Mistral's SSE.
- Neuphonic's Pipecat adapter also uses SSE and yields every chunk immediately (no buffering). Unknown if Neuphonic has the same issue.
- The pre-buffered test uses `task.queue_frame()` which bypasses the TTS service's audio context queue system. The live test goes through `TTSService.tts_process_generator()` → audio context queue → `_handle_audio_context()` → `push_frame()`. These are different code paths.

## File Overview

| File | Purpose |
|------|---------|
| `mistral_tts.py` | The Pipecat TTS adapter (current state with all attempted fixes) |
| `demo_bot.py` | A/B comparison: join the Daily room and listen |
| `diagnostics.py` | Headless transport instrumentation (no listener needed) |
| `.env.example` | API key template |

## What Needs to Happen

Find and fix whatever causes the Daily transport's write timing to go from steady 40ms (pre-buffered) to erratic 7-311ms (live-streamed), given that the audio data is identical and arrives fast enough at every measured point before the transport.
