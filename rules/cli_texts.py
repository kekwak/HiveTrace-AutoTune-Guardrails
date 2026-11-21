import json
import random
import sys

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

REPO, REVISION = "vXofi/guardrails-speed-en-ru", "7324b8823ded971c05076e0c7f4ace5d9b867863"
BUCKETS = {"B1": 512, "B2": 1024, "B3": 2048, "B4": 4096}
N = 400

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
limit = int(sys.argv[2])
extra = int(sys.argv[3]) if len(sys.argv) > 3 else 0
path = hf_hub_download(REPO, "corpus.jsonl", repo_type="dataset", revision=REVISION)
texts = dict.fromkeys(json.loads(line)["messages"][-1]["content"] for line in open(path))
groups = {name: [] for name, top in BUCKETS.items() if top <= limit}
for text in texts:
    length = len(tokenizer(text, truncation=False)["input_ids"]) + extra
    name = next((name for name, top in BUCKETS.items() if length <= top), None)
    if name in groups and text.strip():
        groups[name].append(text)
rng = random.Random(17)
picked = {name: rng.sample(group, min(N, len(group))) for name, group in groups.items()}
json.dump(picked, sys.stdout, ensure_ascii=False)
