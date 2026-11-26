"""Private CLI bridge entry point, executed in the selected locked environment."""

import argparse
import json
import os
import signal
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("eval", "infer"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--baseline-ref")
    args = parser.parse_args()

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    with open(args.config) as stream:
        config = json.load(stream)
    with tempfile.TemporaryDirectory(prefix="u1-runtime-cache-") as cache:
        os.environ.update(
            CUDA_VISIBLE_DEVICES="0",
            TORCHINDUCTOR_CACHE_DIR=str(Path(cache) / "inductor"),
            TRITON_CACHE_DIR=str(Path(cache) / "triton"),
            CUDA_CACHE_PATH=str(Path(cache) / "cuda"),
        )
        from u1_benchmark.cli_integration import evaluate, infer_session

        if args.command == "eval":
            evaluate(config, args.output, args.baseline_ref)
        else:
            # Native libraries must never write into the JSONL response channel.
            output = os.fdopen(os.dup(1), "w", buffering=1)
            os.dup2(2, 1)
            with output, tempfile.TemporaryDirectory(prefix="u1-native-") as directory:
                infer_session(config, sys.stdin, output, directory)


if __name__ == "__main__":
    main()
