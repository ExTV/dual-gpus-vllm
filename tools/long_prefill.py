"""Long-context stress: one prompt of ~N tokens with a hidden code word in the middle.

Reports the time to answer, whether the code word came back, the peak VRAM of every GPU
(sampled once a second with nvidia-smi) and the preemption counter. Run it before trusting
any context / VRAM setting: a config that boots can still run out of memory on its first
long request.

    python tools/long_prefill.py <model_dir_for_tokenizer> [n_tokens] [base_url] [served_model_name]

A 260K prompt takes ~2.7 minutes and vLLM logs nothing while it runs; that is not a hang.
"""
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request

from transformers import AutoTokenizer

TOK_DIR = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 258000
BASE = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8888"
NAME = sys.argv[4] if len(sys.argv) > 4 else "qwen3.8-27b"

tok = AutoTokenizer.from_pretrained(TOK_DIR)
filler = "The quick brown fox jumps over the lazy dog while the river runs quietly past the old mill and the market opens at dawn. "
ids = tok.encode(filler * (N // 24), add_special_tokens=False)[:N - 400]
needle = f"The secret code word is PELICAN-7734. Run {time.time():.0f}."
text = tok.decode(ids[: len(ids) // 2]) + "\n" + needle + "\n" + tok.decode(ids[len(ids) // 2:])

peaks, stop = {}, threading.Event()


def sample():
    while not stop.is_set():
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used",
                                           "--format=csv,noheader,nounits"]).decode()
            for line in out.strip().splitlines():
                idx, name, used = [x.strip() for x in line.split(",")]
                peaks[f"{idx} {name}"] = max(peaks.get(f"{idx} {name}", 0), int(used))
        except (OSError, subprocess.CalledProcessError) as e:
            print("nvidia-smi failed:", e)
        time.sleep(1)


threading.Thread(target=sample, daemon=True).start()
body = {"model": NAME, "max_tokens": 32, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": text + "\n\nWhat is the secret code word? Answer with the code word only."}]}
t0 = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(
    BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"}), timeout=3600))
dt = time.time() - t0
time.sleep(2)
stop.set()
answer = r["choices"][0]["message"]["content"]
print(f"prompt {r['usage']['prompt_tokens']} tokens, answered in {dt:.1f} s: {answer!r} "
      f"({'correct' if 'PELICAN-7734' in answer else 'WRONG'})")
for gpu, mib in sorted(peaks.items()):
    print(f"peak VRAM GPU {gpu}: {mib} MiB")
m = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
pre = re.search(r"vllm:num_preemptions_total\{[^}]*\} (\S+)", m)
print("preemptions:", pre.group(1) if pre else "n/a")
