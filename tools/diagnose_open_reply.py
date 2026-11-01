"""Inspect one native OpenGuardrails reply without changing its scoring contract."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--all-smoke-inputs", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from u1_benchmark.environment import probe
    probe(args.environment, args.output)
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from u1_benchmark.adapters import check_index, load, parse_generated
    from u1_benchmark.datasets import load_speed
    from u1_benchmark.messages import payload
    from u1_benchmark.worker_v2 import current_rows

    data, _ = load_speed(args.dataset, "open", "smoke")
    rows = current_rows("smoke", data)
    adapter = None
    with torch.inference_mode():
        try:
            adapter = load("open", {"backend": "eager"}, args.output, input_version=3)
            report = {"diagnostic_only": True, "quality_evaluated": False,
                      "contract": adapter.contract, "max_new_tokens": 10, "cases": []}
            cases = {row["index"]["total_tokens"]: (row["id"], payload(row), row["index"]) for row in rows}
            if args.all_smoke_inputs:
                edge, _ = adapter.boundary()
                cases[adapter.limit] = ("diagnostic_exact_boundary", edge, adapter.index(edge))
            else:
                cases = dict(list(cases.items())[:1])
            for key, value, index in cases.values():
                check_index(adapter.index(value), index)
                inputs = adapter.prepare(value)
                with sdpa_kernel(SDPBackend.MATH):
                    output = adapter.model.generate(**inputs, do_sample=False, max_new_tokens=10)
                ids = output[0, inputs["input_ids"].shape[-1]:].cpu().tolist()
                text = adapter.tokenizer.decode(ids, skip_special_tokens=True)
                case = {"id": key, "index": index, "output_ids": ids,
                        "raw_output": adapter.tokenizer.decode(ids, skip_special_tokens=False), "decoded_output": text}
                try:
                    case["parsed"] = parse_generated(text, "open", len(ids))
                except ValueError as error:
                    case["parse_error"] = str(error)
                report["cases"].append(case)
                (args.output / "reply.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
                print(json.dumps({"id": key, "total_tokens": index["total_tokens"], "raw_output": case["raw_output"],
                                  "parse_error": case.get("parse_error")}, ensure_ascii=False), flush=True)
                if "parse_error" in case:
                    break
        finally:
            if adapter is not None:
                adapter.close()


if __name__ == "__main__":
    main()
