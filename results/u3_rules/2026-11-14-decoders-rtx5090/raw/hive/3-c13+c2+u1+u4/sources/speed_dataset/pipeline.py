"""Index, validate and freeze shared bucket intersections; no benchmark runner."""

import json
import math
import platform
import traceback
from collections import Counter
from pathlib import Path

from .core import (CONTRACTS, QUOTAS, SEED, digest, file_hash, preparation_hash, implementation_hash,
                   validate_index, validate_pool, write_json, write_jsonl)


def index_pool(directory, model, reuse=None):
    from .preparation import Preparer
    directory = Path(directory)
    rows, pool = validate_pool(directory)
    output = directory / "indices"
    output.mkdir(exist_ok=True)
    meta = {"model": model, "pool_sha256": pool["pool_sha256"],
            "contract_sha256": digest(CONTRACTS["models"][model]),
            "preparation_sha256": preparation_hash(), "status": "incomplete",
            "python": platform.python_version(), "quality_evaluated": False}
    write_json(output / f"{model}.meta.json", meta)
    try:
        p = Preparer(model)
        meta.update(versions=p.versions, assets=p.assets, contract=p.spec, parity=p.parity())
        from .preparation import GUIDELINE, QUESTIONS
        meta["input_contract"] = {
            "prompt": getattr(p, "prompt", None),
            "chat_template": getattr(p, "template", None) or getattr(getattr(p, "tokenizer", None), "chat_template", None),
            "guideline": GUIDELINE if model == "shield" else None,
            "questions": QUESTIONS if model == "laya" else None,
            "schema": p.schema.to_dict() if model == "gliner" else None,
            "cue": "Safety: " if model == "halo" else None,
            "truncation": False, "padding": False,
        }
        cached = {}
        if reuse is not None:
            old_pool, old_manifest = validate_pool(reuse)
            cached, old_meta = validate_index(Path(reuse) / "indices", model, old_pool, old_manifest)
            if old_meta["assets"] != p.assets or old_meta["versions"] != p.versions:
                raise ValueError("Cannot reuse index: tokenizer assets/environment changed")
            meta["reuse"] = {"pool_sha256": old_manifest["pool_sha256"],
                             "index_sha256": old_meta["index_sha256"],
                             "rows": sum(row["id"] in cached for row in rows)}
        write_json(output / f"{model}.meta.json", meta)
        # Keep completed rows if interrupted; .partial is never accepted by finalize.
        partial = output / f"{model}.jsonl.partial"
        with partial.open("w") as stream:
            for i, row in enumerate(rows):
                item = cached[row["id"]] if row["id"] in cached else p.index(row)
                stream.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")
                if (i + 1) % 1000 == 0:
                    stream.flush()
                    print(f"{model}: {i + 1}/{len(rows)}", flush=True)
        partial.replace(output / f"{model}.jsonl")
        meta.update(status="complete", count=len(rows), index_sha256=file_hash(output / f"{model}.jsonl"))
        write_json(output / f"{model}.meta.json", meta)
    except BaseException as error:
        meta.update(status="not_ready", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        write_json(output / f"{model}.meta.json", meta)
        raise


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    x = (len(values) - 1) * q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def describe(values, low, high):
    width = max(1, math.ceil((high - low + 1) / 8))
    hist = [{"min": a, "max": min(a + width - 1, high),
             "count": sum(a <= n <= min(a + width - 1, high) for n in values)}
            for a in range(low, high + 1, width)]
    return {"n": len(values), "min": min(values) if values else None,
            "median": quantile(values, .5), "p95": quantile(values, .95),
            "max": max(values) if values else None, "histogram": hist}


def select(pool, indices, used=None, quotas=None):
    used = set() if used is None else used
    quotas = QUOTAS if quotas is None else quotas
    results = {}
    for bucket, spec in CONTRACTS["buckets"].items():
        missing = [m for m in spec["models"] if m not in indices]
        present = [m for m in spec["models"] if m in indices]
        counts, rejects = Counter(), {m: Counter() for m in present}
        accepted = []
        for row in pool:
            valid = True
            for model in present:
                item = indices[model][row["id"]]
                if not item["supported"] or item["bucket"] != bucket:
                    valid = False
                    rejects[model][item.get("reason") or "different_bucket"] += 1
            if valid and row["id"] not in used:
                accepted.append(row)
                counts[(row["language"], row["source"])] += 1
        shortage = {f"{lang}/{source}": max(0, n - counts[(lang, source)])
                    for (lang, source), n in quotas.items()}
        result = {"status": "not_ready" if missing or any(shortage.values()) else "ready",
                  "missing_models": missing, "intersection_provisional": bool(missing),
                  "intersection_count": len(accepted), "strata_available": {"/".join(k): v for k, v in counts.items()},
                  "shortage": shortage, "rejections": rejects}
        if result["status"] == "ready":
            selected = []
            for (lang, source), n in quotas.items():
                eligible = [r for r in accepted if r["language"] == lang and r["source"] == source]
                eligible.sort(key=lambda r: digest([SEED, bucket, r["id"]]))
                selected.extend(eligible[:n])
            selected.sort(key=lambda r: digest([SEED, "order", bucket, r["id"]]))
            result["ids"] = [r["id"] for r in selected]
            used.update(result["ids"])
            result["composition"] = {"/".join(k): v for k, v in Counter((r["language"], r["source"]) for r in selected).items()}
            result["formats"] = dict(Counter(r["format"] for r in selected))
            result["lengths"] = {m: describe([indices[m][r["id"]]["total_tokens"] for r in selected], spec["min"], spec["max"])
                                 for m in spec["models"]}
        results[bucket] = result
    return results


def finalize(directory):
    directory = Path(directory)
    pool, manifest = validate_pool(directory)
    indices, index_meta, blockers = {}, {}, {}
    for model in CONTRACTS["models"]:
        try:
            indices[model], index_meta[model] = validate_index(directory / "indices", model, pool, manifest)
        except (ValueError, KeyError, OSError, TypeError) as error:
            blockers[model] = str(error)
    results = select(pool, indices)
    status = "ready" if all(r["status"] == "ready" for r in results.values()) else "not_ready"
    fingerprint = digest({"pool": manifest["pool_sha256"], "contracts": CONTRACTS,
                          "indices": {m: digest(v) for m, v in index_meta.items()},
                          "preparation": preparation_hash(), "implementation": implementation_hash()})
    dest = directory / "releases" / fingerprint[:16]
    dest.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        if result["status"] == "ready":
            write_json(dest / f"{name}.ids.json", result["ids"])
    selected_ids = {i for result in results.values() for i in result.get("ids", [])}
    write_jsonl(dest / "corpus.jsonl", [row for row in pool if row["id"] in selected_ids])
    report = {"status": status, "version": CONTRACTS["version"], "fingerprint": fingerprint,
              "pool_sha256": manifest["pool_sha256"], "pool_count": len(pool), "selected_count": len(selected_ids),
              "seed": SEED, "quality_evaluated": False, "gpu_verified": False,
              "contracts": CONTRACTS, "indices": index_meta, "blockers": blockers, "buckets": results,
              "files": {p.name: file_hash(p) for p in dest.glob("*.json*") if p.name != "manifest.json"}}
    write_json(dest / "manifest.json", report)
    lines = ["# Общий корпус скорости", "", f"Статус: **{status}**. Выбрано {len(selected_ids)} из {len(pool)} текстов.",
             "", "CPU-подготовка входов; веса не загружались. GPU не проверен. Разметки качества нет.",
             "Распределение EN/RU и источников задано экспериментатором, не production-трафиком.", ""]
    for name, result in results.items():
        lines += [f"## {name}: {result['status']}", ""]
        if result["status"] != "ready":
            lines += ["```json", json.dumps(result, ensure_ascii=False, indent=2), "```", ""]
            continue
        lines += [f"Пересечение: {result['intersection_count']}; выбрано: {len(result['ids'])}.", "",
                  "Модель | N | min | median | p95 | max", "--- | ---: | ---: | ---: | ---: | ---:"]
        for model, stat in result["lengths"].items():
            lines.append(f"{model} | {stat['n']} | {stat['min']} | {stat['median']} | {stat['p95']} | {stat['max']}")
        lines += ["", "Гистограммы, состав и причины отказов: manifest.json.", ""]
    lines += ["## Ограничения", "", "Границы бакета целиком не покрываются. Выводы относятся только к выбранным текстам.",
              "Ноль изменений вердикта не доказывает сохранение safety. Нужен отдельный размеченный набор.",
              "Wikipedia: CC-BY-SA-3.0/GFDL; URL, авторство и позиции фрагмента указаны в каждой записи.",
              "Синтетические тексты: CC0-1.0. Готовые бакеты не делают весь комплект ready при отсутствии B1."]
    (dest / "REPORT.md").write_text("\n".join(lines) + "\n")
    write_json(directory / "latest.json", {"release": str(dest.relative_to(directory)), "status": status,
                                          "manifest_sha256": file_hash(dest / "manifest.json")})
    print(json.dumps({"status": status, "release": str(dest), "selected": len(selected_ids),
                      "blockers": blockers}, ensure_ascii=False, indent=2), flush=True)
    return report


def import_index(directory, archive, model="llama"):
    """Validate the colleague's complete response before replacing a local index."""
    import tempfile
    import zipfile
    directory = Path(directory)
    pool, manifest = validate_pool(directory)
    names = {f"{model}.jsonl", f"{model}.meta.json"}
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        with zipfile.ZipFile(archive) as z:
            if len(z.namelist()) != 2 or set(z.namelist()) != names:
                raise ValueError("Response must contain only model.jsonl and model.meta.json")
            for name in names:
                if z.getinfo(name).file_size > 100 * 1024 * 1024:
                    raise ValueError("Unexpectedly large index")
                (temp / name).write_bytes(z.read(name))
        validate_index(temp, model, pool, manifest)
        (directory / "indices").mkdir(exist_ok=True)
        for name in sorted(names):
            target = directory / "indices" / name
            pending = target.with_suffix(target.suffix + ".tmp")
            pending.write_bytes((temp / name).read_bytes())
            pending.replace(target)
    print(f"Validated and imported: {model}")
