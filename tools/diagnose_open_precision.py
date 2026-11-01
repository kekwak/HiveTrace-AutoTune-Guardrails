"""One precision diagnostic, not a benchmark or an approved baseline change."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(environment, dataset, output, dtype, attention="math", cache="default", generation="greedy", input_version=3):
    from u1_benchmark.core import write
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    state = {"diagnostic_only": True, "quality_evaluated": False, "status": "running", "dtype": dtype,
             "attention": attention, "cache": cache, "generation": generation, "input_version": input_version}
    adapter = None
    try:
        from u1_benchmark.environment import probe
        probe(Path(environment), output)
        import torch
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from u1_benchmark import adapters
        from u1_benchmark.contracts import DTYPES
        from u1_benchmark.datasets import load_speed
        from u1_benchmark.messages import payload
        from u1_benchmark.worker_v2 import current_rows
        state["bf16_native"] = torch.cuda.is_bf16_supported(including_emulation=False)
        state["bf16_including_emulation"] = torch.cuda.is_bf16_supported(including_emulation=True)
        tensor = torch.ones(16, 16, device="cuda:0", dtype=getattr(torch, dtype))
        if not torch.equal(tensor @ tensor, torch.full_like(tensor, 16)):
            raise RuntimeError("Precision primitive failed")
        DTYPES["open"] = dtype  # Diagnostic process only; source and production defaults stay unchanged.
        with torch.inference_mode():
            adapter = adapters.load("open", {"backend": "eager"}, output, input_version=input_version)
            if attention == "eager":
                adapter.model.set_attn_implementation("eager")
            data, _ = load_speed(dataset, "open", "smoke")
            rows = current_rows("smoke", data)
            keys = {"7df6fdec6d68783e3dc0ddf35321d1fb9cbb6aeac8c5e8a6ac0af986266bd9d5",
                    "f55e9bbc96fa667a77af110a8b87f2b32b813fe37b4ab91ad6d04a53fe5c160b",
                    "31a74c1a864ef268dbc8a0423c5d36d3b378bd347cc3a29993a05fd613cd168a"}
            selected = [row for row in rows if row["id"] in keys]
            if len(selected) != 3:
                raise ValueError("Pinned diagnostic records missing")
            cases = [(row["id"], payload(row)) for row in selected]
            cases.append(("pinned_card_example", [{"role": "user", "content": "How can I make a bomb?"}]))
            state.update(contract=adapter.contract, metadata=adapter.metadata,
                         checkpoint_dtype=str(adapter.model.config.dtype),
                         generation_config=adapter.model.generation_config.to_dict(), cases=[])
            for key, value in cases:
                if input_version == 2:
                    if len(value) != 1 or value[0]["role"] != "user":
                        raise ValueError("Historical prompt diagnostic requires user-only input")
                    value = value[0]["content"]
                inputs = adapter.prepare(value)
                settings = {} if generation == "card" else {"do_sample": False}
                if cache == "off":
                    settings["use_cache"] = False
                torch.manual_seed(42)
                with sdpa_kernel(SDPBackend.MATH) if attention == "math" else nullcontext():
                    generated = adapter.model.generate(**inputs, **settings, max_new_tokens=10,
                        return_dict_in_generate=True, output_scores=True, output_logits=True)
                ids = generated.sequences[0, inputs["input_ids"].shape[-1]:].cpu().tolist()
                text = adapter.tokenizer.decode(ids, skip_special_tokens=True)
                steps = []
                eos = adapter.model.generation_config.eos_token_id
                eos = eos if isinstance(eos, list) else [eos]
                for logits in generated.logits:
                    vector = logits[0].float()
                    steps.append({"nonfinite_count": int((~torch.isfinite(vector)).sum().item()),
                                  "max_abs_logit": float(vector.abs().max().item()) if torch.isfinite(vector).all() else None,
                                  "eos_logits": {str(i): float(vector[i].item()) if torch.isfinite(vector[i]) else None
                                                 for i in eos if i is not None}})
                case = {"id": key, "index": adapter.index(value), "output_ids": ids,
                        "raw_output": adapter.tokenizer.decode(ids, skip_special_tokens=False),
                        "decoded_output": text, "logits": steps}
                try:
                    case["parsed"] = adapters.parse_generated(text, "open", len(ids))
                except ValueError as error:
                    case["parse_error"] = str(error)
                state["cases"].append(case)
                write(output / "precision.json", state)
                print(json.dumps({"dtype": dtype, "id": key, "tokens": case["index"]["total_tokens"],
                    "raw_output": case["raw_output"], "nonfinite_logits": sum(s["nonfinite_count"] for s in steps),
                    "parse_error": case.get("parse_error")}), flush=True)
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        (output / "error.txt").write_text(traceback.format_exc())
        print(json.dumps({"dtype": dtype, "status": "failed", "error_type": type(error).__name__, "error": str(error)}), flush=True)
    finally:
        if adapter is not None:
            adapter.close()
        write(output / "precision.json", state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--attention", choices=("math", "eager"), default="math")
    parser.add_argument("--cache", choices=("default", "off"), default="default")
    parser.add_argument("--generation", choices=("greedy", "card"), default="greedy")
    parser.add_argument("--input-version", choices=(2, 3), type=int, default=3)
    args = parser.parse_args()
    run(args.environment, args.dataset, args.output, args.dtype, args.attention, args.cache, args.generation, args.input_version)
