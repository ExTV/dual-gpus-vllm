"""Decode step time at long context.

Builds one ~N-token prompt with a needle in the middle, sends it twice (the second hits the
prefix cache, so its TTFT is the cache restore, not a prefill) and reports, for each pass:
TTFT, decode tok/s after the first token, ms per draft step (from /metrics draft deltas),
acceptance, and whether the needle came back.

    venv/bin/python tools/long_decode.py <model_dir> [n_tokens] [max_tokens] [base_url] [name]
"""
import json
import re
import sys
import time
import urllib.request

from transformers import AutoTokenizer

TOK_DIR = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 250000
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 300
BASE = sys.argv[4] if len(sys.argv) > 4 else "http://127.0.0.1:8888"
NAME = sys.argv[5] if len(sys.argv) > 5 else "qwen3.8-27b"
KEYS = ("spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total")


def metrics():
    m = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    return {k: sum(float(v) for v in re.findall(r"vllm:" + k + r"\{[^}]*\} (\S+)", m)) for k in KEYS}


tok = AutoTokenizer.from_pretrained(TOK_DIR)
filler = "The quick brown fox jumps over the lazy dog while the river runs quietly past the old mill and the market opens at dawn. "
ids = tok.encode(filler * (N // 24), add_special_tokens=False)[:N - 600]
needle = f"The secret code word is PELICAN-7734. Run {int(time.time())}."
half = len(ids) // 2
text = tok.decode(ids[:half]) + "\n" + needle + "\n" + tok.decode(ids[half:])
prompt = (text + "\n\nFirst, state the secret code word from the text above. Then write a detailed "
          "Python module implementing an LRU cache class with get, put, delete and resize, with docstrings.")


def run(label):
    body = {"model": NAME, "stream": True, "max_tokens": MAXTOK, "temperature": 0,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    m0 = metrics()
    t0, first, usage, out = time.time(), None, None, []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {}).get("content") or ""
                if delta:
                    if first is None:
                        first = time.time()
                    out.append(delta)
    t1 = time.time()
    m1 = metrics()
    steps = m1[KEYS[0]] - m0[KEYS[0]]
    drafted = m1[KEYS[1]] - m0[KEYS[1]]
    acc = m1[KEYS[2]] - m0[KEYS[2]]
    ntok = usage["completion_tokens"] if usage else 0
    dec = t1 - first if first else 0
    txt = "".join(out)
    print(f"{label}: prompt_tokens={usage['prompt_tokens'] if usage else '?'} ttft={first - t0:.1f}s "
          f"out={ntok} decode={dec:.1f}s -> {(ntok - 1) / dec if dec else 0:.1f} tok/s, "
          f"{1000 * dec / steps if steps else 0:.1f} ms/step, steps={steps:.0f}, "
          f"accept={100 * acc / drafted if drafted else 0:.1f}%, needle={'OK' if 'PELICAN-7734' in txt else 'MISS'}")
    return txt


run("cold")
run("cached")
