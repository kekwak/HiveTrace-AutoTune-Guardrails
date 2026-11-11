"""Within-VM noise diagnostic and scenario-local pairing. Not a significance test."""

import csv
from collections import defaultdict
from pathlib import Path
import statistics
import time

from .common import compare_blocks, paired_schedule, write_json

PROTOCOL = "scenario-paired-aa-v1"
AA_BLOCKS, AA_REPEATS = 2, 30


def comparisons(path, baseline, candidate):
    groups = defaultdict(list)
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            groups[(row["scenario"], row["boundary"], row["batch_size"])].append(row)
    result = []
    for key, rows in groups.items():
        pairs = defaultdict(lambda: defaultdict(list))
        for row in rows:
            pairs[(row["session_id"], row["block"])][row["variant"]].append(row)
        for pair in pairs.values():
            a, b = pair.get(baseline, []), pair.get(candidate, [])
            identity = lambda r: (r["iteration"], r["input_hash"], r["total_tokens"], r["padded_tokens"])
            if not a or [identity(r) for r in a] != [identity(r) for r in b]:
                raise ValueError("Paired inputs/order changed")
        result.append({"scenario": key[0], "boundary": key[1], "batch_size": int(key[2]),
                       **compare_blocks(rows, baseline, candidate)})
    return result


def assess_noise(aa, ab, summaries, baseline):
    """Conservative screening flags; no claimed confidence from two A/A pairs."""
    index = {(r["scenario"], r["boundary"], r["batch_size"]): r for r in aa}
    results = []
    for row in ab:
        key = (row["scenario"], row["boundary"], row["batch_size"])
        diagnostic = index.get(key)
        blocks = [b for session in row["sessions"].values() for b in session["blocks"]]
        ratio = statistics.median(b["p50_ratio"] for b in blocks)
        reasons = []
        aa_bound = None
        if diagnostic is None:
            reasons.append("missing_aa")
        else:
            aa_ratios = [b["p50_ratio"] for s in diagnostic["sessions"].values() for b in s["blocks"]]
            # Symmetric multiplicative discrepancy, so A/A slot order does not change the envelope.
            aa_bound = max(max(r, 1 / r) - 1 for r in aa_ratios)
            if 1 - ratio <= aa_bound:
                reasons.append("effect_not_larger_than_observed_aa")
        controls = [s["p50_ms"] for s in summaries if s["variant"] == baseline and
                    (s["scenario"], s["boundary"], s["batch_size"]) == key]
        drift = max(controls) / min(controls) - 1 if controls else None
        if drift is None:
            reasons.append("missing_control_drift")
        elif drift >= 1 - ratio:
            reasons.append("effect_not_larger_than_control_drift")
        windows = defaultdict(list)
        for summary in summaries:
            if (summary["scenario"], summary["boundary"], summary["batch_size"]) == key and "started_unix" in summary:
                windows[summary["block"]].append((summary["started_unix"] + summary["finished_unix"]) / 2)
        gaps = {str(block): abs(values[1] - values[0]) for block, values in windows.items() if len(values) == 2}
        if not all(b["p50_ratio"] < 1 for b in blocks):
            reasons.append("inconsistent_effect_direction")
        if row["p50_ratio_interval_95"][1] >= 1:
            reasons.append("block_interval_includes_no_improvement")
        if any(b["p95_ratio"] > 1.05 for b in blocks):
            reasons.append("p95_regression")
        if ratio > .9:
            reasons.append("below_10_percent_target")
        results.append({"scenario": key[0], "boundary": key[1], "batch_size": key[2],
                        "candidate_control_p50_ratio": ratio, "observed_aa_envelope": aa_bound,
                        "control_p50_range_relative_to_min": drift,
                        "pair_midpoint_gaps_seconds": gaps,
                        "status": "inconclusive" if reasons else "promising_requires_confirmation",
                        "reasons": reasons})
    return {"protocol": PROTOCOL, "results": results, "confirmed_speedup": False,
            "note": "Heuristic screening diagnostic, not a statistical noise bound; A/A precedes A/B and cannot exclude later drift"}


def run_pairs(config, cases, directory, session_id, control_name, control, candidate_name, candidate):
    """Only one GPU model alive. No profiler or full regression inserted between pair members."""
    import torch
    from . import runner
    from .evidence import finish_ort
    directory = Path(directory)
    noise_dir = directory / "noise"
    noise_dir.mkdir()
    write_json(noise_dir / "status.json", {"status": "in_progress", "protocol": PROTOCOL})
    write_json(directory / "measurement_protocol.json", {
        "protocol": PROTOCOL, "pairing": "block -> scenario -> A/B or B/A",
        "aa_blocks": AA_BLOCKS, "aa_repeats": AA_REPEATS, "ab_repeats": 100,
        "warmups_per_boundary": 20, "profiles": "before A/A, in regression directories",
        "warning": "Reload/compile/export separates pair members; inspect timestamps and setup durations"})

    def visit(name, spec, block, scenario, output, repeats):
        target = output / name / str(block) / scenario
        target.mkdir(parents=True)
        adapter = None
        try:
            runner.configure_seed()
            started, setup_timer = time.time(), time.perf_counter()
            adapter = runner.load_adapter(config["model"], spec, target)
            runner.apply_variant(adapter, spec, target)
            torch.cuda.synchronize()
            write_json(target / "metadata.json", {**adapter.metadata, "spec": spec,
                       "setup_started_unix": started, "load_and_patch_ms": (time.perf_counter() - setup_timer) * 1000})
            if hasattr(adapter, "ort_session"):
                finish_ort(adapter, target, cases[scenario][0])
            return runner.measure(adapter, {scenario: cases[scenario]}, output, session_id,
                                  name, block, config["session_seed"], repeats=repeats)
        finally:
            if adapter is not None:
                adapter.close()
            # Each scenario exports afresh. Keep graph inventories/profiles, not 21 copies of weights.
            if spec.get("onnx"):
                removed = []
                for path in target.iterdir():
                    if path.is_file() and not path.is_symlink() and (path.suffix in (".onnx", ".data") or ".onnx." in path.name):
                        removed.append(path.name)
                        path.unlink()
                write_json(target / "export_cleanup.json", {"removed_weight_artifacts": removed,
                           "note": "Regenerable exported weights only; graph inventory and runtime evidence retained"})

    aa_schedule = paired_schedule(AA_BLOCKS, config["session_seed"])
    write_json(noise_dir / "schedule.json", aa_schedule)
    for block, order in enumerate(aa_schedule):
        for scenario in cases:
            for letter in order:
                visit("A1" if letter == "A" else "A2", control, block, scenario, noise_dir, AA_REPEATS)
    aa = comparisons(noise_dir / "latencies.csv", "A1", "A2")
    write_json(noise_dir / "paired_comparisons.json", aa)
    write_json(noise_dir / "status.json", {"status": "completed", "protocol": PROTOCOL,
                                          "quality_evaluated": False, "noise_is_not_a_failure_gate": True})

    summaries = []
    blocks = 3 if config["phase"] == "screening" else 5
    for block, order in enumerate(paired_schedule(blocks, config["session_seed"])):
        for scenario in cases:
            for letter in order:
                name, spec = (control_name, control) if letter == "A" else (candidate_name, candidate)
                summaries += visit(name, spec, block, scenario, directory, 100)
    ab = comparisons(directory / "latencies.csv", control_name, candidate_name)
    write_json(directory / "noise_assessment.json", assess_noise(aa, ab, summaries, control_name))
    return summaries
