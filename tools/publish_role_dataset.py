"""Allowlisted role-corpus export; upload only with the explicit --publish flag."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from u1_benchmark.core import ROOT, MANIFEST_SHA, read, sha, write
from u1_benchmark.contracts import MODEL_SPECS
from u1_benchmark.datasets import load_speed, verified


def export(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    release = read(ROOT / "configs/data/role_speed_v4.json")
    _, manifest = verified(source, release["manifest_sha256"])
    if manifest.get("status") != "ready" or manifest.get("schema_version") != 3:
        raise ValueError("Only the validated native role corpus can be exported")
    for model in MODEL_SPECS:
        load_speed(source, model, "screening", release["manifest_sha256"])
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("manifest.json", *manifest["files"]):
        origin = (source / name).resolve()
        if not origin.is_relative_to(source) or any(p.startswith(".") for p in Path(name).parts):
            raise ValueError("Unexpected corpus path")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
    for target, template in (("README.md", "README_v4.md"), ("LICENSE.md", "LICENSE_v4.md")):
        shutil.copyfile(ROOT / "docs/hf_dataset" / template, destination / target)
    verified(destination, release["manifest_sha256"])
    return destination


def publish(directory, record):
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete, hf_hub_download
    directory, record = Path(directory).resolve(), Path(record)
    release = read(ROOT / "configs/data/role_speed_v4.json")
    _, manifest = verified(directory, release["manifest_sha256"])
    api, repo = HfApi(), release["repo"]
    refs = api.list_repo_refs(repo, repo_type="dataset")
    tags = {r.name: r.target_commit for r in refs.tags}
    if not tags.get("v3"):
        raise ValueError("Historical v3 tag is missing")
    tag_manifest = Path(hf_hub_download(repo, "manifest.json", repo_type="dataset", revision=tags["v3"]))
    if sha(tag_manifest) != MANIFEST_SHA:
        raise ValueError("Historical v3 corpus differs; stop without updating it")
    if "v4" in tags:
        raise ValueError("v4 already exists; verify it rather than moving the tag")
    parent = api.dataset_info(repo).sha
    old_path = Path(hf_hub_download(repo, "manifest.json", repo_type="dataset", revision=parent))
    old_hash = sha(old_path)
    if old_hash not in (MANIFEST_SHA, release["manifest_sha256"]):
        raise ValueError("Unexpected current HF corpus; stop rather than replacing unrelated work")
    names = {"manifest.json", "README.md", "LICENSE.md", *manifest["files"]}
    if old_hash == release["manifest_sha256"]:
        # A previous interrupted publication may have committed data before tagging.
        commit = parent
    else:
        obsolete = set(read(old_path)["files"]) - names
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in obsolete):
            raise ValueError("Unexpected old manifest paths")
        operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(directory / name)) for name in sorted(names)]
        operations += [CommitOperationDelete(path_in_repo=name) for name in sorted(obsolete)]
        result = api.create_commit(repo, repo_type="dataset", parent_commit=parent,
            commit_message="Publish v4 native role corpus with model-specific speed buckets", operations=operations)
        commit = result.oid
    state = {"repo": repo, "revision": commit, "previous_head": parent, "tag": "v4", "tag_created": False,
             "manifest_sha256": release["manifest_sha256"], "historical_v3_revision": api.dataset_info(repo, revision="v3").sha,
             "historical_v3_tag_object": tags["v3"],
             "historical_v3_unchanged": True}
    write(record, state)
    published = Path(hf_hub_download(repo, "manifest.json", repo_type="dataset", revision=commit))
    if sha(published) != release["manifest_sha256"]:
        raise ValueError("Published manifest mismatch")
    api.create_tag(repo, tag="v4", revision=commit, repo_type="dataset", tag_message="Native role speed corpus; CPU validated, no quality labels")
    refs = api.list_repo_refs(repo, repo_type="dataset")
    tags_after = {r.name: r.target_commit for r in refs.tags}
    # HF annotated-tag objects have their own SHA; resolve the file commit, not that object.
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    resolved_v4 = get_hf_file_metadata(hf_hub_url(repo, "manifest.json", repo_type="dataset", revision="v4")).commit_hash
    if tags_after.get("v3") != tags["v3"] or resolved_v4 != commit:
        raise ValueError("Tag verification failed")
    state.update(status="published", tag_created=True)
    write(record, state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    directory = export(args.source, args.output)
    if args.publish:
        if args.record is None:
            parser.error("--publish requires --record for publication diagnostics")
        print(json.dumps(publish(directory, args.record)))
    else:
        print(json.dumps({"status": "exported", "path": str(directory), "published": False}))
