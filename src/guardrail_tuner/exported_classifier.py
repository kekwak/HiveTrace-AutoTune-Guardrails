"""Execute the exact classifier artifact retained by the historical optimizer."""

import hashlib
from pathlib import Path
from typing import Any


def inventory(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            result[str(path.relative_to(root))] = hasher.hexdigest()
    return result


def save_artifacts(model: Any, ort_model: Any, tokenizer: Any, root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    output = {}
    for name, backend in (("pytorch", model), ("onnxruntime", ort_model)):
        target = root / name
        backend.save_pretrained(target)
        tokenizer.save_pretrained(target)
        output[name] = {"path": str(target.resolve()), "inventory": inventory(target)}
    return output


class ExportedClassifier:
    def __init__(self, config: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        spec = config["variant"]
        root = Path(spec["weights"]["path"])
        if not root.is_dir() or inventory(root) != spec["weights"]["inventory"]:
            raise ValueError("Exported artifact is missing or its inventory changed")
        self.preparation, self.scoring = spec["preparation"], spec["scoring"]
        execution = spec["execution"]
        self.device = execution["device"]
        self.runtime = execution["runtime"]
        self.batch_size = config["batch_size"]
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        if self.runtime == "onnxruntime":
            import onnxruntime as ort
            from optimum.onnxruntime import ORTModelForSequenceClassification

            providers = execution["providers"]
            if any(p not in ort.get_available_providers() for p in providers):
                raise ValueError("Requested export provider is unavailable")
            options = ort.SessionOptions()
            options.graph_optimization_level = {
                "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            }[execution["graph_optimization"]]
            options.intra_op_num_threads = execution["intra_op_threads"]
            self.model = ORTModelForSequenceClassification.from_pretrained(
                root,
                export=False,
                provider=providers[0],
                session_options=options,
                provider_options=execution["provider_options"].get(providers[0], {}),
                local_files_only=True,
            )
            self.model.model.disable_fallback()
            if self.model.model.get_providers() != providers:
                raise ValueError("Export provider placement configuration changed")
        elif self.runtime == "pytorch":
            if self.device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable")
            self.model = (
                AutoModelForSequenceClassification.from_pretrained(root, local_files_only=True)
                .to(self.device)
                .eval()
            )
        else:
            raise ValueError("Unknown exported classifier runtime")
        self.metadata = {
            "adapter": "exported_sequence_classification",
            "execution": execution,
            "historical_protocol": True,
            "truncation": self.preparation["truncation"],
        }

    def predict(self, inputs: list[Any]) -> list[dict[str, Any]]:
        import torch

        if not inputs or not all(isinstance(x, str) for x in inputs):
            raise ValueError("This historical classifier artifact accepts plain text only")
        outputs = []
        for start in range(0, len(inputs), self.batch_size):
            encoded = self.tokenizer(
                inputs[start : start + self.batch_size],
                padding=True,
                truncation=self.preparation["truncation"],
                max_length=self.preparation["limit"],
                return_tensors="pt",
            )
            if self.runtime == "pytorch":
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**encoded).logits.float().cpu()
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Nonfinite exported classifier output")
            for row in logits:
                method = self.scoring["method"]
                label = (
                    int(row.argmax())
                    if method == "multiclass"
                    else (
                        (row.sigmoid() >= self.scoring["threshold"]).int().tolist()
                        if method == "multilabel"
                        else None
                    )
                )
                outputs.append({"label": label, "logits": row.tolist()})
        return outputs

    def close(self) -> None:
        self.model = None
