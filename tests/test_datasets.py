import contextlib
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from guardrail_tuner.cli import main
from guardrail_tuner.datasets import DatasetSource, load_source


class DatasetTests(unittest.TestCase):
    def test_hub_revision_is_resolved_and_used_for_loading(self):
        calls = []

        class FakeDataset:
            column_names = ["text"]

            def __len__(self):
                return 3

        dataset = FakeDataset()
        datasets = types.ModuleType("datasets")
        datasets.load_dataset = lambda *a, **kw: calls.append((a, kw)) or dataset
        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = lambda: types.SimpleNamespace(
            dataset_info=lambda path, revision: types.SimpleNamespace(sha="commit123")
        )
        with patch.dict(sys.modules, {"datasets": datasets, "huggingface_hub": hub}):
            result, meta = load_source(DatasetSource("hub", "private/data", "test", "main", "en"))
        self.assertIs(result, dataset)
        self.assertEqual(calls[0][1]["revision"], "commit123")
        self.assertEqual(calls[0][1]["split"], "test")
        self.assertEqual(calls[0][1]["name"], "en")
        self.assertEqual(meta["rows"], 3)

    def test_two_sources_and_no_rows_in_output(self):
        dataset = types.SimpleNamespace()
        metadata = {"rows": 1, "columns": ["text"]}
        args = [
            "guardrail-tuner",
            "datasets",
            "--quality-hf",
            "org/quality",
            "--performance-file",
            "speed.jsonl",
        ]
        with (
            patch.object(sys, "argv", args),
            patch("guardrail_tuner.cli.load_source", return_value=(dataset, metadata)) as loader,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(main(), 0)
        self.assertEqual(loader.call_args_list[0].args[0].kind, "hub")
        self.assertEqual(loader.call_args_list[1].args[0].kind, "file")
        self.assertEqual(
            json.loads(output.getvalue()), {"quality": metadata, "performance": metadata}
        )

    def test_local_file_hash_and_options_are_separate(self):
        class FakeDataset:
            column_names = ["text"]

            def __len__(self):
                return 1

        datasets = types.ModuleType("datasets")
        datasets.load_dataset = lambda *a, **kw: FakeDataset()
        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = object
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"datasets": datasets, "huggingface_hub": hub}),
        ):
            path = Path(directory) / "data.jsonl"
            path.write_text('{"text":"private"}\n')
            _, meta = load_source(DatasetSource("file", str(path)))
        self.assertEqual(
            meta["sha256"], "4d793777e720c9a475606c5966ce04ace6a7fc6221d1bbffd6ddf268d4131940"
        )


if __name__ == "__main__":
    unittest.main()


def test_local_file_glob_characters_select_exact_file(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    directory = tmp_path / "dir[1]"
    directory.mkdir()
    target = directory / "data[1].jsonl"
    target.write_text('{"text":"target"}\n')
    (directory / "data1.jsonl").write_text('{"text":"wrong"}\n{"text":"wrong"}\n')

    dataset, metadata = load_source(DatasetSource("file", str(target)))

    assert len(dataset) == 1
    assert dataset[0]["text"] == "target"
    assert metadata["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()


def test_performance_preset_is_default_and_can_be_overridden(monkeypatch):
    from guardrail_tuner.datasets import SPEED_PRESET

    dataset = types.SimpleNamespace()
    metadata = {"rows": 1, "columns": ["text"]}
    captured = io.StringIO()
    with (
        patch.object(sys, "argv", ["guardrail-tuner", "datasets", "--quality-file", "q.jsonl"]),
        patch("guardrail_tuner.cli.load_source", return_value=(dataset, metadata)) as loader,
        contextlib.redirect_stdout(captured),
    ):
        assert main() == 0
    assert loader.call_args_list[1].args[0] == DatasetSource("preset", SPEED_PRESET)

    with (
        patch.object(
            sys,
            "argv",
            [
                "guardrail-tuner",
                "datasets",
                "--quality-file",
                "q.jsonl",
                "--performance-hf",
                "custom/data",
            ],
        ),
        patch("guardrail_tuner.cli.load_source", return_value=(dataset, metadata)) as loader,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        assert main() == 0
    assert loader.call_args_list[1].args[0] == DatasetSource("hub", "custom/data")

    with (
        patch.object(
            sys,
            "argv",
            [
                "guardrail-tuner",
                "datasets",
                "--quality-file",
                "q.jsonl",
                "--performance-file",
                "p.jsonl",
            ],
        ),
        patch("guardrail_tuner.cli.load_source", return_value=(dataset, metadata)) as loader,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        assert main() == 0
    assert loader.call_args_list[1].args[0] == DatasetSource("file", "p.jsonl")


def test_preset_verifies_manifest_corpus_and_preserves_messages(tmp_path, monkeypatch):
    import guardrail_tuner.datasets as module

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("fixture")
    digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "status": "ready",
                "count": 12000,
                "files": {"corpus.jsonl": digest},
            }
        )
    )
    monkeypatch.setattr(
        module, "SPEED_MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    calls = []

    class FakeDataset:
        column_names = ["id", "messages"]

        def __len__(self):
            return 12000

        def __getitem__(self, key):
            assert key == "messages"
            return [
                [{"role": "user", "content": "private"}, {"role": "assistant", "content": "answer"}]
            ] * 12000

        def add_column(self, key, values):
            assert key == "text" and len(values) == 12000 and values[0] == "answer"
            self.column_names = ["id", "messages", "text"]
            return self

    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: FakeDataset()
    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = object

    def download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(tmp_path / filename)

    hub.hf_hub_download = download
    with patch.dict(sys.modules, {"datasets": datasets, "huggingface_hub": hub}):
        dataset, meta = load_source(DatasetSource("preset", module.SPEED_PRESET))
        assert len(dataset) == 12000
        assert meta["corpus_sha256"] == digest
        assert meta["model_specific_u1_buckets_applied"] is False
        assert "private" not in json.dumps(meta)
        assert all(call[2]["revision"] == module.SPEED_REVISION for call in calls)
        corpus.write_text("tampered")
        with pytest.raises(ValueError, match="checksum mismatch"):
            load_source(DatasetSource("preset", module.SPEED_PRESET))
