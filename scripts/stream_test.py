#!/usr/bin/env python3
import argparse
import http.client
import json
import time
import wave


def post_json(host, port, path, body=b"", headers=None):
    headers = headers or {}
    conn = http.client.HTTPConnection(host, port, timeout=60)
    conn.request("POST", path, body=body, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    if resp.status >= 400:
        raise RuntimeError(f"HTTP {resp.status}: {data.decode('utf-8', errors='ignore')}")
    return json.loads(data.decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Stream a WAV file to the realtime server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3004)
    parser.add_argument("--wav", default="/home/ptthang/wslRealtimeASR/third_party/whisper.cpp/samples/jfk.wav")
    parser.add_argument("--chunk-ms", type=int, default=500)
    parser.add_argument("--realtime", action="store_true", help="Sleep between chunks to simulate realtime")
    args = parser.parse_args()

    session = post_json(args.host, args.port, "/session")
    session_id = session["session_id"]
    print(f"Session: {session_id}")

    with wave.open(args.wav, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise ValueError("WAV must be mono, 16-bit, 16kHz PCM")
        chunk_frames = int(16000 * (args.chunk_ms / 1000.0))
        while True:
            frames = wf.readframes(chunk_frames)
            if not frames:
                break
            resp = post_json(
                args.host,
                args.port,
                "/chunk",
                body=frames,
                headers={"Content-Type": "application/octet-stream", "X-Session-Id": session_id},
            )
            if resp.get("updated"):
                print(resp.get("text", ""))
            if args.realtime:
                time.sleep(args.chunk_ms / 1000.0)

    final = post_json(args.host, args.port, "/stop", headers={"X-Session-Id": session_id})
    print("\nFinal transcript:\n")
    print(final.get("text", ""))


if __name__ == "__main__":
    main()
