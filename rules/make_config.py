import hashlib
import json
import sys
from pathlib import Path

model, rules = sys.argv[1], sys.argv[2].split(",")
mode = sys.argv[3] if len(sys.argv) > 3 else "smoke"
source = Path("rules/adapter.py")
cfg = json.loads(Path(f"configs/benchmark-v3/{model}_baseline.json").read_text())
cfg["mode"] = mode
cfg["candidate"] = {
    "module": str(source),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "dependencies": {},
    "options": {"rules": rules},
}
print(json.dumps(cfg, indent=2))
