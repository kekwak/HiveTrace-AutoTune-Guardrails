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
