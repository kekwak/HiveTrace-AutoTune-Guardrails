"""Отчёты: reports/<label>/ для одной модели, reports/<A>__vs__<B>/ для сравнения. Только локально."""

import json
import math
from pathlib import Path

import pandas as pd

from .spec import ROOT, run_label

REPORTS_DIR = ROOT / "reports"


def _f(x, pct=False, digits=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{100 * x:.1f}%" if pct else f"{x:.{digits}f}"


def _ci(row, m, pct=True):
    return f"{_f(row[m], pct)} [{_f(row[f'{m}_lo'], pct)}; {_f(row[f'{m}_hi'], pct)}]"


def _md(df):
    if df.empty:
        return "_нет данных_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def _quality_view(t, keys):
    return pd.DataFrame([{**{k: r[k] for k in keys}, "N": r["n"], "harm/safe": f"{r['n_harm']}/{r['n_safe']}",
                          "FNR": _ci(r, "fnr"), "FPR": _ci(r, "fpr"), "macro-F1": _ci(r, "macro_f1", False),
                          "ROC-AUC": _f(r["roc_auc"])} for _, r in t.iterrows()])


def write_single(run, tables, origin, spec):
    out = REPORTS_DIR / run_label(spec)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        if isinstance(t, pd.DataFrame):
            t.to_csv(out / f"{name}.csv", index=False)
    o = tables["overall"]
    lines = [f"# {run_label(spec)}\n", f"Модель `{spec['model_id']}`, quant `{spec.get('quant')}`, dtype `{spec['dtype']}`, "
             f"порог {spec['threshold']}. Прогон `{run.dir.relative_to(ROOT)}`. Датасет: `{json.dumps(origin, ensure_ascii=False)}`. "
             f"Строк с ошибкой: {tables['errors']}.\n",
             "## Итог\n", _md(pd.DataFrame([{"агрегат": r["aggregate"], "FNR": _ci(r, "fnr"), "FPR": _ci(r, "fpr"),
                                              "macro-F1": _ci(r, "macro_f1", False)} for _, r in o.iterrows()])),
             "## По бенчмаркам\n", _md(_quality_view(tables["benchmark"], ["benchmark"])),
             "## По сплитам\n", _md(_quality_view(tables["split"], ["benchmark", "split"])),
             "## По хвостам\n", _md(_quality_view(tables["tail"], ["tail"]))]
    (out / "summary.md").write_text("\n".join(lines))
    return out


def _compare_view(t, keys):
    return pd.DataFrame([{**{k: r[k] for k in keys}, "N": r["n"],
                          "смен": f"{r['flips']} ({_f(r['flip_rate'], True)} [{_f(r['flip_rate_lo'], True)}; {_f(r['flip_rate_hi'], True)}])",
                          "safe→harm / harm→safe": f"{r['safe_to_harm']} / {r['harm_to_safe']}",
                          "хуже / лучше": f"{r['worse']} / {r['better']}", "McNemar p": _f(r["mcnemar_p"]),
                          "ΔFNR": _ci(r, "delta_fnr"), "ΔFPR": _ci(r, "delta_fpr"),
                          "Δmacro-F1": _ci(r, "delta_macro_f1", False)} for _, r in t.iterrows()])


def _speed_table(sa, sb):
    if not (sa and sb and sa.get("status") == "complete" and sb.get("status") == "complete"):
        return None
    rows = []
    for name in sa["buckets"]:
        a, b = sa["buckets"][name], sb["buckets"].get(name)
        if not b:
            continue
        rows.append({"бакет": name, "токенов": a["user_tokens_median"],
                     "p50 A, мс": _f(a["p50_ms"], digits=2), "p50 B, мс": _f(b["p50_ms"], digits=2),
                     "p95 A, мс": _f(a["p95_ms"], digits=2), "p95 B, мс": _f(b["p95_ms"], digits=2),
                     "снижение p95": _f(1 - b["p95_ms"] / a["p95_ms"], True),
                     "пик памяти A/B, МБ": f"{_f(a['peak_allocated_mb'], digits=0)} / {_f(b['peak_allocated_mb'], digits=0)}"})
    return pd.DataFrame(rows)


def write_compare(run_a, run_b, tables, rows, origin, spec_a, spec_b, top=30):
    out = REPORTS_DIR / f"{run_label(spec_a)}__vs__{run_label(spec_b)}"
    out.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        if isinstance(t, pd.DataFrame):
            t.to_csv(out / f"{name}.csv", index=False)
    thr, thr_b = spec_a["threshold"], spec_b["threshold"]
    flipped = rows[(rows["p_a"] > thr) != (rows["p_b"] > thr_b)].copy()
    flipped["abs_dp"] = (flipped["p_a"] - flipped["p_b"]).abs()
    flipped.sort_values("abs_dp", ascending=False).to_csv(out / "flips.csv", index=False)

    qa, qb = run_a.manifest().get("quant", {}), run_b.manifest().get("quant", {})
    speed = _speed_table(run_a.speed(), run_b.speed())
    o = tables["overall"]
    lines = [f"# {run_label(spec_a)} (A) vs {run_label(spec_b)} (B)\n",
             f"Порог A {thr}, B {thr_b}. Датасет: `{json.dumps(origin, ensure_ascii=False)}`. Исключено строк с ошибкой: {tables['excluded_rows']}.\n",
             f"- A: `{run_a.dir.relative_to(ROOT)}`, quant `{spec_a.get('quant')}`, веса {_f((qa.get('weight_bytes') or float('nan')) / 2**20, digits=0)} МБ",
             f"- B: `{run_b.dir.relative_to(ROOT)}`, quant `{spec_b.get('quant')}`, веса {_f((qb.get('weight_bytes') or float('nan')) / 2**20, digits=0)} МБ, "
             f"квантовано слоёв {qb.get('quantized_linear', '—')}\n",
             "## Итог: изменённые вердикты B относительно A\n",
             _md(pd.DataFrame([{"агрегат": r["aggregate"], "смен": "—" if pd.isna(r.get("flips")) else int(r["flips"]),
                                "доля смен": f"{_f(r['flip_rate'], True)} [{_f(r['flip_rate_lo'], True)}; {_f(r['flip_rate_hi'], True)}]",
                                "ΔFNR": _ci(r, "delta_fnr"), "ΔFPR": _ci(r, "delta_fpr"),
                                "Δmacro-F1": _ci(r, "delta_macro_f1", False)} for _, r in o.iterrows()])),
             "## По бенчмаркам\n", _md(_compare_view(tables["benchmark"], ["benchmark"])),
             "## По сплитам\n", _md(_compare_view(tables["split"], ["benchmark", "split"])),
             "## По хвостам\n", _md(_compare_view(tables["tail"], ["tail"])),
             "## По типу атаки\n", _md(_compare_view(tables["attack_type"], ["benchmark", "attack_type"]))]
    if speed is not None:
        lines += ["## Скорость (отдельные прогоны, без парного A/B)\n", _md(speed)]
    else:
        lines += ["## Скорость\n", "_нет замеров скорости для обоих прогонов: запустите `python -m guard_eval speed`_\n"]
    lines += [f"## Изменённые вердикты (первые {top} по |Δp|; полный список — flips.csv)\n",
              _md(flipped.sort_values("abs_dp", ascending=False).head(top)
                  .assign(text=lambda d: d["text"].str.slice(0, 120).str.replace("\n", " ").str.replace("|", "/"))
                  [["benchmark", "split", "label", "p_a", "p_b", "text"]].round(3))]
    (out / "summary.md").write_text("\n".join(lines))

    pooled, macro = o.iloc[0], o.iloc[1]
    sa, sb = run_a.speed(), run_b.speed()
    headline = {"a": run_label(spec_a), "b": run_label(spec_b), "model": spec_b["name"], "quant_b": spec_b.get("quant"),
                "n": int(pooled["n"]), "flips": int(pooled["flips"]), "flip_rate": pooled["flip_rate"],
                "flip_rate_ci": [pooled["flip_rate_lo"], pooled["flip_rate_hi"]], "worse": int(pooled["worse"]),
                "better": int(pooled["better"]), "macro_flip_rate": macro["flip_rate"],
                "macro_delta_fnr": macro["delta_fnr"], "macro_delta_fpr": macro["delta_fpr"],
                "weights_mb": [(qa.get("weight_bytes") or 0) / 2**20, (qb.get("weight_bytes") or 0) / 2**20],
                "p95_reduction": ({k: 1 - sb["buckets"][k]["p95_ms"] / v["p95_ms"] for k, v in sa["buckets"].items()
                                   if k in sb.get("buckets", {})}
                                  if sa and sb and sa.get("status") == sb.get("status") == "complete" else None)}
    (out / "headline.json").write_text(json.dumps(headline, ensure_ascii=False, indent=2, default=float))
    return out


def write_overview():
    """reports/SUMMARY.md — сводка всех сравнений: доля смен вердиктов и снижение p95."""
    rows = []
    for path in sorted(REPORTS_DIR.glob("*__vs__*/headline.json")):
        h = json.loads(path.read_text())
        red = h.get("p95_reduction") or {}
        rows.append({"модель": h["model"], "A": h["a"], "B": h["b"], "N": h["n"],
                     "смен": f"{h['flips']} ({_f(h['flip_rate'], True)} [{_f(h['flip_rate_ci'][0], True)}; {_f(h['flip_rate_ci'][1], True)}])",
                     "хуже/лучше": f"{h['worse']}/{h['better']}", "macro смен": _f(h["macro_flip_rate"], True),
                     "macro ΔFNR": _f(h["macro_delta_fnr"], True), "macro ΔFPR": _f(h["macro_delta_fpr"], True),
                     "веса A→B, МБ": f"{h['weights_mb'][0]:.0f}→{h['weights_mb'][1]:.0f}",
                     **{f"p95 ↓ {k}": _f(red.get(k), True) for k in ("varying_short", "varying_medium", "varying_long", "mixed")}})
    out = REPORTS_DIR / "SUMMARY.md"
    REPORTS_DIR.mkdir(exist_ok=True)
    out.write_text("# Сводка сравнений\n\n" + _md(pd.DataFrame(rows)) +
                   "\nсмен — изменённые вердикты B относительно A (pooled, 95% bootstrap CI); p95 ↓ — снижение p95 полного запроса.\n")
    return out, len(rows)
