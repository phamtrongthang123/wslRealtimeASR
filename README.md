# Realtime Whisper.cpp Web Dictation

Realtime dictation webapp powered by `whisper.cpp` with CUDA support. The browser streams 16 kHz PCM audio chunks to the backend, which performs incremental transcription and returns the rolling transcript.

## Prerequisites

- Conda installed
- NVIDIA driver installed (for CUDA)

## Setup

```bash
./scripts/setup_conda.sh
conda activate wslrealtimeasr
./scripts/download_model.sh small.en
./scripts/build.sh
```

## Run

```bash
./scripts/run.sh
```

Open `http://127.0.0.1:3004` in your browser and click **Start recording**.

## Systemd service

Install the unit (requires sudo):

```bash
sudo cp /home/ptthang/wslRealtimeASR/systemd/whisper-realtime.service /etc/systemd/system/whisper-realtime.service
sudo systemctl daemon-reload
sudo systemctl enable --now whisper-realtime.service
```

Check status/logs:

```bash
systemctl status whisper-realtime.service
journalctl -u whisper-realtime.service -f
```

## Smoke test

With the server running in another terminal:

```bash
conda activate wslrealtimeasr
python scripts/stream_test.py --realtime
```

## Server options

```bash
./build/realtime_server --help
```

Common overrides:

- `--model /path/to/ggml-*.bin`
- `--port 3004`
- `--step-ms 1000` (processing cadence)
- `--window-ms 8000` (context window)
- `--commit-lag-ms 2000` (how long to wait before locking text)
- `--no-gpu` to force CPU
- `--keep-context` to reuse prompt context between chunks (off by default)
- `--no-context` to force disable prompt reuse
- `--no-fallback` to disable temperature fallback (looping prevention)
- `--max-repeat 2` maximum consecutive identical sentences to keep (0 disables filtering)

## Notes

- The frontend sends mono 16 kHz PCM chunks. If you use a custom client, keep the same format.
- For CPU-only builds, rerun the build with `-DGGML_CUDA=OFF`.
- Browsers only allow mic access on HTTPS or `http://localhost`. If you open the UI via a LAN IP, use HTTPS (reverse proxy) or an SSH tunnel.
