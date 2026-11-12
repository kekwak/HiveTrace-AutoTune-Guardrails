import pytest

from guardrail_tuner.workload import Workload, batches


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(int(text)))}


def test_partition_token_lengths_and_record_truncation():
    plan = Workload((4, 8), (1, 2))
    grouped, truncated = plan.partition(["2", "4", "6", "11"], Tokenizer(), 10)
    assert grouped == {"4": ["2", "4"], "8": ["6"], ">8": ["11"]}
    assert truncated == 1
    assert batches(grouped["4"], 2) == [["2", "4"]]


@pytest.mark.parametrize(
    "buckets,batch_sizes", [((4, 4), (1,)), ((0, 4), (1,)), ((4,), (2,)), ((4,), (1, 1))]
)
def test_invalid_workload_rejected(buckets, batch_sizes):
    with pytest.raises(ValueError):
        Workload(buckets, batch_sizes)
