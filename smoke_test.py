#!/usr/bin/env python3
"""Minimal check that the shim is up and a checkpoint responds.
Usage: uv run python smoke_test.py [PORT] [MODEL_KEY_or_tinker_path]
"""
import sys, json, urllib.request, pathlib
port = sys.argv[1] if len(sys.argv) > 1 else "8000"
reg = json.loads((pathlib.Path(__file__).parent / "models.json").read_text())["checkpoints"]
model = sys.argv[2] if len(sys.argv) > 2 else "step-856"
model = reg.get(model, model)  # allow a friendly key or a raw tinker:// path
body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "In one sentence, what is reward hacking?"}],
    "max_tokens": 200, "temperature": 1.0,
}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                            data=body, headers={"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=300).read())
msg = r["choices"][0]["message"]
print("model:", r["model"])
print("reasoning:", (msg.get("reasoning_content") or "")[:300])
print("content:", msg["content"])
print("usage:", r["usage"])
