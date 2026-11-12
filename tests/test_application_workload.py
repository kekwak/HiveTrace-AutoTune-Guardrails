from guardrail_tuner.application_workload import profile_application
from guardrail_tuner.workload import Workload


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text)))}


class Runner:
    def __init__(self, bad=False):
        self.bad = bad

    def run(self, value):
        if "inputs" in value:
            return {
                "outputs": [
                    {"value": -1 if self.bad else len(row["text"])} for row in value["inputs"]
                ]
            }
        return {"value": len(value["text"])}


def test_jsonl_batch_workload_validates_against_individual_results():
    records = [{"text": "abc"}, {"text": "abcd"}]

    def matches(a, b):
        return a["value"] == b["value"]

    result = profile_application(
        Runner(),
        Runner(),
        Runner(),
        records,
        records,
        Tokenizer(),
        "text",
        Workload((4,), (1, 2)),
        0,
        matches,
    )
    assert [row["status"] for row in result["results"]] == ["preliminary", "preliminary"]
    assert result["results"][1]["quality"]["passed"]
    bad = profile_application(
        Runner(),
        Runner(),
        Runner(True),
        records,
        records,
        Tokenizer(),
        "text",
        Workload((4,), (1, 2)),
        0,
        matches,
    )
    assert all(row["reason"] == "batch_output_mismatch" for row in bad["results"])
