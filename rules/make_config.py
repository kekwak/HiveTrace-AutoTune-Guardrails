import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
from u1_benchmark.v4_preparation import validate_settings

model = sys.argv[1]
rules = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else []
batch = int(sys.argv[3]) if len(sys.argv) > 3 else 1
source = Path("rules/adapter.py")
data = Path("data/speed/rules-v4.json")
cfg = json.loads(Path(f"configs/v4/{model}.json").read_text())
cfg["mode"] = "search" if rules else "baseline"
cfg["batch_sizes_by_bucket"] = {b: [batch] for b in cfg["buckets"]}
if data.exists():
    built = json.loads(data.read_text())
    cfg["data"] = {"path": built["path"], "manifest_sha256": built["manifest_sha256"]}
if rules:
    cfg["candidate"] = validate_settings(cfg["baseline"])
    cfg["candidate"]["adapter"] = {
        "module": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dependencies": {},
    }
    cfg["candidate"]["execution"].update(model=model, rules=rules, batch=batch)
print(json.dumps(cfg, indent=2))
