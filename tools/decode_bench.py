"""Decode speed on natural text (code, prose, reasoning), the way the README numbers were measured.

Streams a chat completion with thinking off at temperature 0, twice per prompt, and reports
decode tok/s (tokens after the first / time after the first) plus the speculative-decoding
acceptance read from /metrics.

    python tools/decode_bench.py [base_url] [served_model_name]
"""
import json
import re
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8888"
NAME = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b"
PROMPTS = {
    "code": "Write a complete Python module implementing an LRU cache class with get, put, delete, resize and a thread-safe variant, with docstrings and a small test suite using unittest.",
    "prose": "Write a long, detailed essay about the history of the printing press and its effects on European society, religion and science. Use several paragraphs.",
    "reason": "A train leaves city A at 9:40 travelling at 80 km/h toward city B, 350 km away. Another leaves B at 10:10 at 95 km/h toward A. Explain step by step when and where they meet, then verify the answer two different ways.",
}
KEYS = ("spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total")


def metrics():
    m = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    return {k: sum(float(v) for v in re.findall(r"vllm:" + k + r"\{[^}]*\} (\S+)", m)) for k in KEYS}


def run(prompt):
    body = {"model": NAME, "stream": True, "max_tokens": 700, "temperature": 0,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, first, usage = time.time(), None, None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            if first is None and d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                first = time.time()
    n = usage["completion_tokens"]
    return n, (n - 1) / (time.time() - first), first - t0


for name, prompt in PROMPTS.items():
    m0 = metrics()
    results = [run(prompt) for _ in range(2)]
    m1 = metrics()
    d = {k: m1[k] - m0[k] for k in KEYS}
    rates = [r[1] for r in results]
    spec = (f"acceptance {d[KEYS[2]] / d[KEYS[1]] * 100:.1f}%, {d[KEYS[2]] / d[KEYS[0]] + 1:.2f} tokens/step"
            if d[KEYS[1]] else "no speculative decoding")
    print(f"{name:6s} {results[-1][0]} tokens  decode {min(rates):.1f}-{max(rates):.1f} tok/s  "
          f"TTFT {results[-1][2]:.2f} s  {spec}")
