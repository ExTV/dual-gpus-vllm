"""Prefix-cache check: are follow-up turns reusing the cache or re-reading the whole conversation?

Sends a ~33K-token conversation cold, repeats it, then extends it twice. With caching working
the repeat and the follow-ups take about a second; if they take as long as the cold request,
every chat turn is being re-prefilled (see docs/how-it-works.md, "fast follow-up turns").

    python tools/cache_check.py [n_words] [base_url] [served_model_name]
"""
import json
import random
import sys
import time
import urllib.request

WORDS = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8888"
NAME = sys.argv[3] if len(sys.argv) > 3 else "qwen3.8-27b"
vocab = ["alpha", "river", "stone", "cloud", "ember", "maple", "orbit", "quartz", "delta", "lumen"]
rnd = random.Random(time.time())


def words(n):
    return " ".join(rnd.choice(vocab) for _ in range(n))


def ask(messages):
    body = {"model": NAME, "messages": messages, "max_tokens": 8, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=1800))
    return time.time() - t, r["usage"]["prompt_tokens"]


m1 = [{"role": "user", "content": f"Session {rnd.random()}\n{words(WORDS)}\nSay ok."}]
m2 = m1 + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "Now say yes. " + words(300)}]
m3 = m2 + [{"role": "assistant", "content": "yes"}, {"role": "user", "content": "Now say done. " + words(3000)}]
for label, msgs in (("cold", m1), ("same prompt again", m1), ("follow-up +~400 tokens", m2), ("follow-up +~3K tokens", m3)):
    dt, n = ask(msgs)
    print(f"{label:24s} {n:>7} tokens  {dt:6.2f} s")
