"""Adapt configured model outputs to the text moderation API."""

import math
from collections.abc import Mapping
from typing import Any
from uuid import uuid4


class Moderation:
    def __init__(self, config: dict[str, Any], model: str, task: str) -> None:
        outputs = config.get("outputs")
        if (
            not isinstance(outputs, dict)
            or not outputs
            or any(
                not isinstance(label, str)
                or not label.strip()
                or label != label.strip()
                or not isinstance(categories, list)
                or not all(
                    isinstance(category, str) and category.strip() for category in categories
                )
                for label, categories in outputs.items()
            )
        ):
            raise ValueError("moderation.outputs must map labels or verdicts to category lists")
        template = config.get("prompt_template")
        if task == "generation" and (
            not isinstance(template, str) or template.count("{input}") != 1
        ):
            raise ValueError("moderation.prompt_template must contain exactly one {input}")
        if task == "sequence_classification" and "prompt_template" in config:
            raise ValueError("moderation.prompt_template is only supported for generation")
        self.outputs: dict[str, list[str]] = outputs
        self.categories = sorted({category for values in outputs.values() for category in values})
        self.template: str | None = template
        self.model = model

    def prepare(self, body: Any) -> list[str]:
        if not isinstance(body, dict):
            raise ValueError("Expected a JSON object")
        if body.get("model", self.model) != self.model:
            raise ValueError("Requested model does not match the served model")
        values = body.get("input")
        if isinstance(values, str):
            values = [values]
        if (
            not isinstance(values, list)
            or not 0 < len(values) <= 128
            or not all(isinstance(text, str) and text.strip() for text in values)
        ):
            raise ValueError("input must be a non-empty string or a list of up to 128 strings")
        if self.template is not None:
            return [self.template.replace("{input}", text) for text in values]
        return values

    def validate_labels(self, labels: Mapping[int, str], count: int) -> list[str]:
        class_labels = [labels.get(i, str(i)) for i in range(count)]
        if any(label not in self.outputs for label in class_labels):
            raise ValueError("moderation.outputs must cover every classification label")
        return class_labels

    def response(self, outputs: list[dict[str, Any]], labels: Mapping[int, str]) -> dict[str, Any]:
        results = []
        for output in outputs:
            verdict = output["label"] if self.template is None else output["text"].strip()
            if verdict not in self.outputs:
                raise ValueError("Model verdict does not match moderation.outputs")
            flagged = self.outputs[verdict]
            categories = {category: category in flagged for category in self.categories}
            scores: dict[str, float] = {}
            if self.template is None:
                logits = output["logits"]
                if not logits or not all(math.isfinite(logit) for logit in logits):
                    raise ValueError("Invalid classification logits")
                class_labels = self.validate_labels(labels, len(logits))
                peak = max(logits)
                weights = [math.exp(logit - peak) for logit in logits]
                total = sum(weights)
                scores = {
                    category: sum(
                        weight
                        for label, weight in zip(class_labels, weights)
                        if category in self.outputs[label]
                    )
                    / total
                    for category in self.categories
                }
            results.append(
                {
                    "flagged": bool(flagged),
                    "categories": categories,
                    "category_scores": scores,
                    "category_applied_input_types": {
                        category: ["text"] for category in self.categories
                    },
                }
            )
        return {"id": f"modr-{uuid4().hex}", "model": self.model, "results": results}
