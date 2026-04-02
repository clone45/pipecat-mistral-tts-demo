# Pipecat Mistral Voxtral TTS Adapter

A Pipecat TTS service for [Mistral's Voxtral TTS API](https://docs.mistral.ai/capabilities/audio/text_to_speech), streaming audio through Daily WebRTC transport.

## Features

- Streams audio via SSE from Mistral's /v1/audio/speech endpoint
- Converts Voxtral's float32 PCM (24kHz) to int16 PCM for Pipecat
- Supports all preset voices and custom cloned voices
- Pipecat TTFB and usage metrics
- Zero added latency -- audio starts playing as soon as the first chunk arrives

## Quick Start

```bash
# Install dependencies
uv sync

# Configure API keys
cp .env.example .env
# Edit .env with your Daily and Mistral API keys

# Run the demo
uv run python demo_bot.py
```

Open the printed Daily room URL in your browser to hear the bot speak.

## Usage in Your Project

```python
from mistral_tts import MistralTTSService

tts = MistralTTSService(
    api_key="your-mistral-api-key",
    voice_id="gb_jane_neutral",  # or a custom voice UUID
)
```

Drop it into any Pipecat pipeline:

```python
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.transports.daily.transport import DailyParams, DailyTransport

transport = DailyTransport(
    room_url=room_url,
    params=DailyParams(
        audio_out_enabled=True,
        audio_out_sample_rate=48000,  # Important: use 48kHz, not 24kHz
    ),
)

pipeline = Pipeline([transport.input(), tts, transport.output()])
task = PipelineTask(
    pipeline,
    params=PipelineParams(audio_out_sample_rate=48000),
)
```

## Important: Transport Sample Rate

Set `audio_out_sample_rate=48000` on both the transport and pipeline params. Pipecat's SOXR resampler automatically converts the 24kHz TTS output to 48kHz for Daily's WebRTC engine. Running the transport at 24kHz causes choppy playback.

## Available Voices

Preset voices include regional variants with different emotions:

- English (US): en_paul_neutral, en_paul_cheerful, en_paul_excited, en_paul_sad, ...
- English (GB): gb_jane_neutral, gb_jane_confident, gb_oliver_cheerful, ...
- French: fr_marie_neutral, fr_marie_happy, fr_marie_curious, ...

You can also create custom voices via the [Mistral Voices API](https://docs.mistral.ai/capabilities/audio/text_to_speech/voices) and use their UUIDs as voice_id.

## Technical Notes

This adapter uses raw `aiohttp` HTTP streaming instead of the `mistralai` Python SDK. The SDK's SSE event parser batches audio into large chunks (up to 96KB) which blocks the Python event loop and starves the Daily WebRTC transport's audio write thread. Reading bytes directly from the HTTP response with `iter_any()` lets the network I/O naturally pace the event loop, producing smooth audio. This matches the pattern used by Pipecat's built-in NeuphonicHttpTTSService.

## Requirements

- Python 3.11+
- pipecat-ai 0.0.108+
- aiohttp
- numpy
- A [Mistral API key](https://console.mistral.ai)
- A [Daily.co API key](https://dashboard.daily.co) (for the demo)

## Files

| File | Description |
|------|-------------|
| mistral_tts.py | The Pipecat TTS adapter -- copy this into your project |
| demo_bot.py | Simple demo bot that speaks when you join |
| .env.example | API key template |
