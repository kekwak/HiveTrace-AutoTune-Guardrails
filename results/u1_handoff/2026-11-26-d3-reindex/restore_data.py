"""Restore the measured input indices onto the pinned public corpus, without models."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from u1_benchmark.core import sha
from u1_benchmark.v4_data import verified

ARCHIVE = Path(__file__).with_name("input-indices.zip")
ARCHIVE_SHA256 = "633e786d4b936a488d6fa18e1f103e741342953e9c920685efad34c4a96930a9"
CORPUS_SHA256 = "4f64ed678c40f758ee96ebdb4c1d10fe154bada937d924ba27580ad3bebafbca"
SNAPSHOTS = {
    "new-families-20261125": "2dce7a829dab27437d607847318fb96e78175f1cf9870d7b013ec3835749aafb",
    "new-families-d3-20261126": "846e234bd0dff69ca1d22d460e49b9c4dfdc85e08576f10f01f46bc0dd5cbc2c",
}


def _members(archive):
    members = {}
    for item in archive.infolist():
        name = item.filename
        parts = name.split("/")
        mode = item.external_attr >> 16
        if ("\\" in name or PurePosixPath(name).is_absolute()
                or any(not part or part.startswith(".") for part in parts)
                or len(parts) < 2 or parts[0] not in SNAPSHOTS
                or name in members or item.is_dir()
                or stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
            raise ValueError("Unsafe or duplicate index archive member")
        members[name] = item
    return members


def restore(source, output_root):
    source, output_root = Path(source).resolve(), Path(output_root).resolve()
    corpus = source / "corpus.jsonl"
    for name in SNAPSHOTS:
        target = output_root / name
        if target.exists() or target.is_symlink():
            raise FileExistsError("Snapshot destination already exists")
    if not corpus.is_file() or sha(corpus) != CORPUS_SHA256:
        raise ValueError("Source corpus checksum mismatch; use the pinned fetch-data corpus")
    if sha(ARCHIVE) != ARCHIVE_SHA256:
        raise ValueError("Index archive checksum mismatch")

    with zipfile.ZipFile(ARCHIVE) as archive:
        members = _members(archive)
        expected = set()
        for name, checksum in SNAPSHOTS.items():
            manifest_name = name + "/manifest.json"
            if manifest_name not in members:
                raise ValueError("Missing snapshot manifest")
            manifest_bytes = archive.read(members[manifest_name])
            if hashlib.sha256(manifest_bytes).hexdigest() != checksum:
                raise ValueError("Snapshot manifest checksum mismatch")
            manifest = json.loads(manifest_bytes)
            if manifest["files"].get("corpus.jsonl") != CORPUS_SHA256:
                raise ValueError("Snapshot corpus checksum mismatch")
            expected.add(manifest_name)
            expected.update(name + "/" + p for p in manifest["files"] if p != "corpus.jsonl")
        if set(members) != expected:
            raise ValueError("Unexpected or missing index archive files")

        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".restore-indices-", dir=output_root) as tmp:
            staging = Path(tmp)
            for name, item in members.items():
                destination = staging / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as src, destination.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
            # Validate both snapshots before publishing either destination.
            for name, checksum in SNAPSHOTS.items():
                shutil.copyfile(corpus, staging / name / "corpus.jsonl")
                verified(staging / name, checksum)
            for name in SNAPSHOTS:
                target = output_root / name
                if target.exists() or target.is_symlink():
                    raise FileExistsError("Snapshot destination already exists")
            for name in SNAPSHOTS:
                (staging / name).rename(output_root / name)
    return {"status": "ready", "snapshots": SNAPSHOTS,
            "corpus_sha256": CORPUS_SHA256, "archive_sha256": ARCHIVE_SHA256,
            "weights_loaded": False, "gpu_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Directory returned by benchmark.py fetch-data")
    parser.add_argument("--output-root", type=Path, default=Path("data/speed"))
    args = parser.parse_args()
    try:
        result = restore(args.source, args.output_root)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
