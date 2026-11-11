"""Audit user artifacts, generate frozen bundles, reports and self-contained notebooks."""

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
from pathlib import Path
import statistics

from .runtime import digest, read, verify_bundle, write

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("legacy_short", "varying_short", "legacy_medium", "varying_medium",
             "legacy_long", "varying_long", "mixed")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def csv_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def audit_run(run, sources):
    manifest = read(run / "manifest.json")
    approval = read(run / "approval.json")
    if manifest["status"] != "completed" or manifest["source_hash"] != digest(sources):
        raise ValueError("Incomplete run or source mismatch: " + str(run))
    if approval["source_hash"] != digest(sources) or not approval["regression_passed"]:
        raise ValueError("Invalid approval")
    raw = csv_rows(run / "latencies.csv")
    groups = defaultdict(list)
    for row in raw:
        groups[(row["scenario"], row["boundary"], row["block"], row["variant"])].append(row)
    if len(raw) != 8400 or len(groups) != 84 or any(len(v) != 100 for v in groups.values()):
        raise ValueError("Incomplete screening blocks")
    if {r["scenario"] for r in raw} != set(SCENARIOS):
        raise ValueError("Scenario set changed")
    cfg = manifest["config"]
    for (scenario, boundary, block, variant), rows in groups.items():
        if variant != cfg["control"]:
            continue
        other = groups[(scenario, boundary, block, cfg["candidate"])]
        identity = lambda row: tuple(row[k] for k in ("iteration", "input_hash", "total_tokens", "padded_tokens"))
        if list(map(identity, rows)) != list(map(identity, other)):
            raise ValueError("Different paired inputs")
    aa = csv_rows(run / "noise" / "latencies.csv")
    if len(aa) != 1680:
        raise ValueError("Incomplete AA measurements")
    regression = {}
    for variant in (cfg["control"], cfg["candidate"]):
        value = read(run / "regression" / variant / "regression.json")
        if not value["passed"] or value["n"] != 300:
            raise ValueError("Development regression missing/failed")
        regression[variant] = value
    traces = []
    for path in sorted(run.rglob("*_manifest.json")):
        meta = read(path)
        if "gz_sha256" not in meta:
            continue
        trace = path.with_name(path.name.removesuffix("_manifest.json") + ".json.gz")
        if sha(trace) != meta["gz_sha256"]:
            raise ValueError("Compressed trace checksum mismatch")
        with gzip.open(trace, "rb") as stream:
            raw_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if raw_hash != meta["raw_sha256"]:
            raise ValueError("Uncompressed trace checksum mismatch")
        traces.append({"path": str(trace.relative_to(run)), **meta, "verified": True})
    if len(traces) < 7:
        raise ValueError("Missing backend/profiling traces")
    return {"run_id": run.name, "source_hash": digest(sources), "status": "verified",
            "ab_rows": len(raw), "aa_rows": len(aa), "regression": regression,
            "traces": traces, "artifact_hashes": {str(p.relative_to(run)): sha(p)
                      for p in sorted(run.rglob("*")) if p.is_file() and p.suffix != ".zip"}}


def report_run(run, audit):
    manifest = read(run / "manifest.json")
    cfg = manifest["config"]
    control, candidate = cfg["control"], cfg["candidate"]
    summary = read(run / "summary.json")
    noise = {r["scenario"]: r for r in read(run / "noise_assessment.json")["results"]
             if r["boundary"] == "text_to_verdict"}
    lines = [f"# {cfg['model']}: {candidate} / {control}", "",
             f"Screening `{run.name}`; источник: `{run}`.",
             "Одна T4-сессия. Это не окончательное подтверждение ускорения и не оценка safety-качества.",
             f"Аудит: {audit['ab_rows']} A/B, {audit['aa_rows']} A/A, {len(audit['traces'])} проверенных trace.",
             "", "## Полный путь: текст → вердикт", "",
             "p50/p95 ниже — медианы трёх блочных квантилей, не квантиль объединённых запросов. "
             "Отношение вычислено внутри пар, поэтому может отличаться от отношения этих медиан.", "",
             "| Сценарий | p50 контроль → кандидат, мс | p95 контроль → кандидат, мс | Отношение p50 | A/A, % | Дрейф контроля, % |",
             "| --- | --- | --- | --- | --- | --- |"]
    for scenario in SCENARIOS:
        subsets = {v: [r for r in summary if r["scenario"] == scenario and
                       r["variant"] == v and r["boundary"] == "text_to_verdict"]
                   for v in (control, candidate)}
        med = lambda v, key: statistics.median(r[key] for r in subsets[v])
        n = noise[scenario]
        lines.append(f"| {scenario} | {med(control, 'p50_ms'):.2f} → {med(candidate, 'p50_ms'):.2f} | "
                     f"{med(control, 'p95_ms'):.2f} → {med(candidate, 'p95_ms'):.2f} | "
                     f"{n['candidate_control_p50_ratio']:.4f} | {100*n['observed_aa_envelope']:.2f} | "
                     f"{100*n['control_p50_range_relative_to_min']:.2f} |")
    lines += ["", "## Память, подготовка и численные ответы", "",
              "Память — максимум Torch allocated/reserved среди серий, включая веса; не полный расход процесса. "
              "Load+patch не включает отложенную компиляцию; прогрев включает исполнение и возможную компиляцию, "
              "поэтому не является чистой стоимостью компиляции.", ""]
    for variant in (control, candidate):
        values = [r for r in summary if r["variant"] == variant]
        metadata = [read(p) for p in (run / variant).glob("*/*/metadata.json")]
        setup = [m["load_and_patch_ms"] for m in metadata]
        reg = audit["regression"][variant]
        lines.append(f"- `{variant}`: allocated/reserved {max(r['peak_allocated_mib'] for r in values):.1f}/"
                     f"{max(r['peak_reserved_mib'] for r in values):.1f} MiB; median load+patch "
                     f"{statistics.median(setup):.1f} мс; median warmup {statistics.median(r['warmup_ms'] for r in values):.1f} мс. "
                     f"Δp max={reg['delta_max']:.8f}, p99={reg['delta_p99']:.8f}, смен метки={len(reg['flips'])}; "
                     f"примеров около порога={reg['near_threshold_count']}.")
    lines += ["", "## Профиль short/long", "",
              "GPU-время под профайлером нельзя вычитать из обычного p50. "
              "Fused kernels показаны целиком: наличие `copy`/`norm` в имени не определяет стоимость компонента.", ""]
    for variant in (control, candidate):
        for length in ("short", "long"):
            base = run / "regression" / variant / "profiles"
            rows = read(base / f"legacy_{length}_attribution.json")["rows"]
            iterations = read(base / f"legacy_{length}_manifest.json")["iterations"]
            categories = defaultdict(lambda: [0, 0.0])
            for row in rows:
                fused = "fused" in row["operator"].lower() or "fused" in row["kernel"].lower()
                label = "fused_composite" if fused else row["component"]
                categories[label][0] += row["kernels"]
                categories[label][1] += row["gpu_us"]
            parts = "; ".join(f"{k}: {v[0]/iterations:.0f} kernels, {v[1]/iterations/1000:.2f} мс"
                              for k, v in sorted(categories.items(), key=lambda x: -x[1][1]))
            lines.append(f"- `{variant}` / {length}: {parts}.")
    lines += ["", "## Вывод", "",
              "Кандидат требует трёх независимых confirmation-сессий. Holdout в этом screening не использовался. "
              "Глобально лучший вариант не установлен; сравнения с другими сочетаниями не выполнены.",
              "Стоимость окупаемости будет рассчитана после подтверждения; из суммарного прогрева нельзя "
              "выделить точную cold-compile стоимость."]
    return "\n".join(lines) + "\n"


def cell(kind, source, identifier):
    value = {"cell_type": kind, "metadata": {}, "id": identifier, "source": source.splitlines(True)}
    if kind == "code":
        value.update(outputs=[], execution_count=None)
    return value


def notebook(bundle, diagnostic=False):
    wrappers = {name: (ROOT / "followup" / name).read_text()
                for name in ("__init__.py", "runtime.py", "onnx_diagnostic.py")}
    title = "GLiNER: диагностика ONNX" if diagnostic else bundle["manifest"]["config"]["candidate"] + ": подтверждение"
    intro = f"""# {title}

Подготовлено и проверено на CPU; этот новый workflow ещё не запускался на T4.
Новая Kaggle-сессия с T4 и Internet. Colab допустим только при совпадении среды.
Исторический стек: Python 3.13.15, PyTorch 2.11.0+cu128. PyTorch автоматически не заменяется.
Локальный клон и загрузка approval не нужны: точный код screening встроен и проверяется по SHA256.
Задайте ссылку/идентификатор нового запуска площадки, подтвердите выделение нового runtime ниже.
Не используйте три перезапуска ядра одной VM как три независимых повтора.
"""
    if diagnostic:
        intro += "\nОдин запуск, максимум 2 часа. Strict → диагностический CUDA+CPU. Нет approval и сравнения скорости.\n"
    else:
        intro += ("\nТри НОВЫЕ сессии: SESSION_SEED 17, 42, 123. Один ноутбук на одну модель, шесть запусков всего.\n"
                  "Holdout 300 → A/A 2×30 → A/B 5×100, семь сценариев и две границы. Максимум 10 часов.\n"
                  "При ошибке/шуме скачайте архив, не повторяйте до удобного результата.\n")
    config = ("SESSION_SEED = 17\nFRESH_RUNTIME_CONFIRMED = False  # True только после выделения нового runtime\n"
              "PROVIDER_RUN = \"\"  # ссылка/ID новой Kaggle-версии или Colab-сеанса, без токенов доступа\n")
    unpack = """import hashlib, json, os, sys, uuid
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['TORCH_LOGS'] = 'graph_breaks,recompiles,perf_hints'
ROOT = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
WORK = ROOT / ('followup_' + uuid.uuid4().hex[:8])
WORK.mkdir()
"""
    unpack += f"BUNDLE = {bundle!r}\nWRAPPERS = {wrappers!r}\n"
    unpack += f"PAYLOAD_SHA256 = {digest({'bundle': bundle, 'wrappers': wrappers})!r}\n"
    unpack += """payload = json.dumps({'bundle': BUNDLE, 'wrappers': WRAPPERS}, sort_keys=True, ensure_ascii=False, allow_nan=False)
if hashlib.sha256(payload.encode()).hexdigest() != PAYLOAD_SHA256:
    raise ValueError('Повреждён payload')
(WORK / 'followup').mkdir()
for name, source in WRAPPERS.items():
    compile(source, name, 'exec')
    (WORK / 'followup' / name).write_text(source)
for name in list(sys.modules):
    if name.split('.')[0] in ('followup', 'experiments'):
        del sys.modules[name]
sys.path.insert(0, str(WORK))
from followup.runtime import restore_bundle, supervise
restore_bundle(BUNDLE, WORK)
print(WORK)
"""
    execute = (f"STATUS = supervise(WORK, SESSION_SEED, FRESH_RUNTIME_CONFIRMED, PROVIDER_RUN, "
               f"total_seconds={7200 if diagnostic else 36000}, diagnostic={diagnostic})\nprint(STATUS)\n")
    finish = """ARCHIVE = WORK.with_suffix('.zip')
print('Скачать, включая ошибки:', ARCHIVE)
try:
    from google.colab import files
except ImportError:
    print('Kaggle: Output → followup_* → ZIP')
else:
    files.download(str(ARCHIVE))
"""
    interpretation = ("## Статусы\n`completed` — этап выполнен, не гарантия ускорения/качества. "
                      "Смотрите status.json, result.json, worker.log. environment_diff.json объясняет несовпадения среды. "
                      "При timeout сохраняются доступные стадии. Удаление VM площадкой может прервать архивацию. "
                      "Диагностика ONNX никогда не даёт performance approval. "
                      "Полный итог подтверждения строится локально после получения трёх архивов каждой модели.\n")
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}, "frozen_source_hash": bundle["approval"]["source_hash"]},
            "cells": [cell("markdown", intro, "intro"), cell("code", config, "config"),
                      cell("code", unpack, "unpack"), cell("code", execute, "run"),
                      cell("markdown", interpretation, "status"), cell("code", finish, "download")]}


def prepare(gliner_root, halo_root):
    audit_index = {}
    bundles = {}
    for model, root, experiment in (("gliner", gliner_root, "gliner_compile"),
                                    ("halo", halo_root, "haloguard_fla")):
        root = Path(root)
        sources = {p.name: p.read_text() for p in (root / "source").glob("*.py")}
        run, = (p for p in (root / experiment / "screening" / "runs").iterdir()
                if p.is_dir() and (p / "manifest.json").is_file())
        audit = audit_run(run, sources)
        audit_index[model] = audit
        bundle = {"sources": sources, "manifest": read(run / "manifest.json"),
                  "approval": read(run / "approval.json"), "environment": read(run / "environment.json"),
                  "install_report": read(root / "environment" / "install_report.json")}
        verify_bundle(bundle)
        bundles[model] = bundle
        write(ROOT / "followup" / "frozen" / (model + ".json"), bundle)
        (ROOT / "docs" / f"screening_{model}_2026-10-26.md").write_text(report_run(run, audit))
    sdpa, = (p for p in (Path(halo_root) / "haloguard_sdpa" / "screening" / "runs").iterdir()
             if p.is_dir() and (p / "manifest.json").is_file())
    audit_index["sdpa"] = audit_run(sdpa, bundles["halo"]["sources"])
    onnx = Path(gliner_root) / "gliner_onnx" / "preflight_native"
    audit_index["onnx"] = {"status": read(onnx / "status.json"),
                           "error": (onnx / "error.txt").read_text(),
                           "log_sha256": sha(onnx.parent / "preflight_native.log")}
    write(ROOT / "docs" / "round2_artifact_audit.json", audit_index)
    for name, bundle, diagnostic in (("haloguard_fla_confirmation", bundles["halo"], False),
                                      ("gliner_compile_confirmation", bundles["gliner"], False),
                                      ("gliner_onnx_diagnostic", bundles["gliner"], True)):
        write(ROOT / "notebooks" / (name + ".ipynb"), notebook(bundle, diagnostic))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gliner", required=True)
    parser.add_argument("--halo", required=True)
    args = parser.parse_args()
    prepare(args.gliner, args.halo)
