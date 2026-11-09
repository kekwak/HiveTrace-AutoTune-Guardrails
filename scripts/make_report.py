"""Собирает отчёт по этапу 2 (динамическое квантование на T4) из архивов guard_eval."""

import glob
import json
import math
import os
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
R_EAGER = BASE / "results/t4/eager"
R_PROFILE = BASE / "results/t4/profile"
R_COMPILE = BASE / "results/t4/compile"
STATS = BASE / "results/t4/tails_v1_stats.csv"
OUT = BASE / "docs/report_quantization_T4.md"

MODELS = ["gliner-guard", "koala", "hivetrace-guard-pro", "openguardrails"]
TITLE = {"gliner-guard": "gliner-guard-omni", "koala": "KoalaAI/Text-Moderation",
         "hivetrace-guard-pro": "HiveTraceGuard-Pro", "openguardrails": "OpenGuardrails-Text-4B-0124"}
KIND = {"gliner-guard": "encoder, 307M (GLiNER2)", "koala": "encoder, 139M (DeBERTa)",
        "hivetrace-guard-pro": "decoder, 0.6B (Qwen3)", "openguardrails": "decoder, 4B (Qwen3)"}
METHODS = ["bnb_int8", "torchao_int8_dyn"]
MNAME = {"bnb_int8": "bnb LLM.int8()", "torchao_int8_dyn": "torchao W8A8"}
BUCKETS = ["legacy_short", "varying_short", "legacy_medium", "varying_medium", "legacy_long", "varying_long", "mixed"]


def isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def pct(x, d=1, sign=False):
    if isnan(x):
        return "—"
    return f"{100 * x:+.{d}f}" if sign else f"{100 * x:.{d}f}"


def num(x, d=1):
    return "—" if isnan(x) else f"{x:.{d}f}"


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out) + "\n"


def rep(root, model, method=None):
    name = f"{model}__none__float16" + (f"__vs__{model}__{method}__float16" if method else "")
    return Path(root) / "reports" / name


def read(path, name):
    p = Path(path) / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def runs(root):
    out = {}
    for m in glob.glob(f"{root}/runs/*/*/manifest.json"):
        d = json.load(open(m))
        s = d["spec"]
        sp = Path(m).parent / "speed.json"
        out[(s["name"], s["quant"], bool(s.get("compile")))] = {
            "manifest": d, "speed": json.load(open(sp)) if sp.exists() else None,
            "pred": Path(m).parent / "predictions.parquet"}
    return out


EAGER, COMPILE = runs(R_EAGER), runs(R_COMPILE)


def bucket(run, b, key):
    sp = (run or {}).get("speed") or {}
    return sp.get("buckets", {}).get(b, {}).get(key)


# ---------------------------------------------------------------- данные
stats = pd.read_csv(STATS)
ds = (stats.groupby("benchmark").apply(lambda g: pd.Series({
    "harm": int(g.loc[g.label == "harm", "n"].sum()), "safe": int(g.loc[g.label == "safe", "n"].sum()),
    "языки": ", ".join(sorted(g["lang"].unique())), "хвост": ", ".join(sorted(g["tail"].unique()))}), include_groups=False)
      .reset_index())
ds_md = md(ds.rename(columns={"benchmark": "бенчмарк"}))
N_TOTAL = int(stats["n"].sum())

# ---------------------------------------------------------------- общая таблица
overall_rows = []
for model in MODELS:
    base_speed = EAGER.get((model, "none", False))
    for method in METHODS:
        o = read(rep(R_EAGER, model, method), "overall").set_index("aggregate")
        p, m = o.loc["pooled"], o.loc["macro_by_benchmark"]
        run = EAGER.get((model, method, False))
        a95, b95 = bucket(base_speed, "mixed", "p95_ms"), bucket(run, "mixed", "p95_ms")
        am, bm = bucket(base_speed, "mixed", "peak_allocated_mb"), bucket(run, "mixed", "peak_allocated_mb")
        overall_rows.append({
            "модель": TITLE[model], "метод": MNAME[method],
            "смен вердиктов": f"{int(p.flips)} ({pct(p.flip_rate)}% [{pct(p.flip_rate_lo)}; {pct(p.flip_rate_hi)}])",
            "хуже / лучше": f"{int(p.worse)} / {int(p.better)}",
            "macro ΔFNR, п.п.": pct(m.delta_fnr, sign=True), "macro ΔFPR, п.п.": pct(m.delta_fpr, sign=True),
            "Δ macro-F1, п.п.": "—" if isnan(p.delta_macro_f1) else f"{100 * p.delta_macro_f1:+.2f}",
            "p95 mixed, мс": f"{num(a95)} → {num(b95)}",
            "пик памяти, МБ": f"{num(am, 0)} → {num(bm, 0)}"})
overall_md = md(pd.DataFrame(overall_rows))

TOKNOTE = {"gliner-guard": "", "koala": "",
           "hivetrace-guard-pro": " Токены — текст пользователя; шаблон с политикой добавляет ещё ~420 токенов.",
           "openguardrails": " Токены — текст пользователя; инструкция классификатора добавляет ещё ~290 токенов."}
mem = []
for model in MODELS:
    a = bucket(EAGER.get((model, "none", False)), "mixed", "peak_allocated_mb")
    for method in METHODS:
        b = bucket(EAGER.get((model, method, False)), "mixed", "peak_allocated_mb")
        mem.append(1 - b / a)
MEM_RANGE = f"−{100 * min(mem):.0f}…{100 * max(mem):.0f}%"

worst = []
for model in [m for m in MODELS if m != "koala"]:
    for method in METHODS:
        c = read(rep(R_EAGER, model, method), "split")
        c["delta_err"] = c["delta_fnr"].where(c["fnr_a"].notna(), c["delta_fpr"])
        c["metric"] = c["fnr_a"].notna().map({True: "FNR", False: "FPR"})
        for r in c.sort_values("delta_err", ascending=False).head(3).itertuples():
            base = r.fnr_a if r.metric == "FNR" else r.fpr_a
            worst.append({"модель": TITLE[model], "метод": MNAME[method], "сплит": f"{r.benchmark}/{r.split}", "N": int(r.n),
                          "ошибка": r.metric, "fp16 → INT8, %": f"{pct(base)} → {pct(base + r.delta_err)}",
                          "Δ, п.п.": pct(r.delta_err, sign=True), "смен, %": pct(r.flip_rate)})
worst_md = md(pd.DataFrame(worst))

# ---------------------------------------------------------------- качество эталонов по бенчмаркам
bench_rows = []
benchmarks = ["hivetrace_input", "hivetrace_input_robust", "rtplx", "multijail", "csrt"]
for model in MODELS:
    b = read(rep(R_EAGER, model), "benchmark").set_index("benchmark")
    row = {"модель": TITLE[model]}
    for bench in benchmarks:
        r = b.loc[bench]
        cell = f"FNR {pct(r.fnr)}"
        if not isnan(r.fpr):
            cell += f" / FPR {pct(r.fpr)} / F1 {num(r.macro_f1, 2)}"
        row[bench] = cell
    bench_rows.append(row)
bench_md = md(pd.DataFrame(bench_rows))


# ---------------------------------------------------------------- по сплитам для каждой модели
def split_table(model):
    base = read(rep(R_EAGER, model), "split").set_index(["benchmark", "split"])
    cmp = {m: read(rep(R_EAGER, model, m), "split").set_index(["benchmark", "split"]) for m in METHODS}
    rows = []
    for (bench, split), r in base.iterrows():
        harm = r.n_harm > 0
        metric = "fnr" if harm else "fpr"
        row = {"бенчмарк": bench, "сплит": split, "N": int(r.n), "ошибка": metric.upper(),
               "эталон fp16, %": pct(r[metric])}
        for m in METHODS:
            c = cmp[m].loc[(bench, split)]
            d = c[f"delta_{metric}"]
            mark = " ⚠" if not isnan(d) and d >= 0.01 else ""
            row[f"{MNAME[m]}: смен, %"] = f"{pct(c.flip_rate)} ({int(c.worse)}↓/{int(c.better)}↑)"
            row[f"{MNAME[m]}: Δ ошибки, п.п."] = pct(d, sign=True) + mark
        rows.append(row)
    return md(pd.DataFrame(rows))


def tail_table(model):
    rows = []
    for m in METHODS:
        t = read(rep(R_EAGER, model, m), "tail").set_index("tail")
        for tail in ["in_domain", "obfuscation", "paraphrase", "non_en", "en_reference"]:
            if tail not in t.index:
                continue
            r = t.loc[tail]
            rows.append({"метод": MNAME[m], "хвост": tail, "N": int(r.n), "смен, %": pct(r.flip_rate),
                         "хуже / лучше": f"{int(r.worse)} / {int(r.better)}",
                         "ΔFNR, п.п.": pct(r.delta_fnr, sign=True), "ΔFPR, п.п.": pct(r.delta_fpr, sign=True)})
    return md(pd.DataFrame(rows))


# ---------------------------------------------------------------- аугментации hivetrace_input_robust
aug_rows = {}
for model in [m for m in MODELS if m != "koala"]:
    for m in METHODS:
        a = read(rep(R_EAGER, model, m), "attack_type")
        a = a[a.benchmark == "hivetrace_input_robust"].set_index("attack_type")
        for aug, r in a.iterrows():
            aug_rows.setdefault(aug, {"аугментация": aug, "N": int(r.n)})[f"{TITLE[model].split('-')[0]} / {'bnb' if m == 'bnb_int8' else 'torchao'}"] = pct(r.flip_rate)
aug_md = md(pd.DataFrame(list(aug_rows.values())).sort_values("аугментация"))


# ---------------------------------------------------------------- скорость по бакетам
def speed_table(model):
    rows = []
    base = EAGER.get((model, "none", False))
    for b in BUCKETS:
        if bucket(base, b, "p50_ms") is None:
            continue
        row = {"бакет": b, "токенов польз.": bucket(base, b, "user_tokens_median"),
               "fp16 p50 / p95": f"{num(bucket(base, b, 'p50_ms'))} / {num(bucket(base, b, 'p95_ms'))}"}
        for m in METHODS:
            run = EAGER.get((model, m, False))
            a95, b95 = bucket(base, b, "p95_ms"), bucket(run, b, "p95_ms")
            row[f"{MNAME[m]} p50 / p95"] = f"{num(bucket(run, b, 'p50_ms'))} / {num(b95)}"
            row[f"{MNAME[m]}: Δp95"] = pct(b95 / a95 - 1, 0, sign=True) + "%"
        rows.append(row)
    return md(pd.DataFrame(rows))


# ---------------------------------------------------------------- compile (HiveTraceGuard)
HTG = "hivetrace-guard-pro"
conf = [("fp16 eager", EAGER.get((HTG, "none", False))), ("fp16 + compile", COMPILE.get((HTG, "none", True))),
        ("INT8 torchao + compile", COMPILE.get((HTG, "torchao_int8_dyn", True))),
        ("INT8 torchao eager", EAGER.get((HTG, "torchao_int8_dyn", False))),
        ("INT8 bnb eager", EAGER.get((HTG, "bnb_int8", False)))]
comp_rows = []
ref = conf[0][1]
for b in BUCKETS:
    row = {"бакет": b}
    for name, run in conf:
        p95 = bucket(run, b, "p95_ms")
        d = "" if run is ref else f" ({pct(p95 / bucket(ref, b, 'p95_ms') - 1, 0, sign=True)}%)"
        row[name] = f"{num(bucket(run, b, 'p50_ms'))} / {num(p95)}{d}"
    comp_rows.append(row)
comp_rows.append({"бакет": "пик памяти, МБ", **{n: num(bucket(r, "mixed", "peak_allocated_mb"), 0) for n, r in conf}})
compile_speed_md = md(pd.DataFrame(comp_rows))
c2 = {k: bucket(COMPILE.get((HTG, "none", True)), k, "p95_ms") for k in BUCKETS}
c3 = {k: bucket(COMPILE.get((HTG, "torchao_int8_dyn", True)), k, "p95_ms") for k in BUCKETS}
e1 = {k: bucket(EAGER.get((HTG, "none", False)), k, "p95_ms") for k in BUCKETS}
COMPILE_GAIN = [1 - c2[k] / e1[k] for k in ("varying_short", "varying_medium", "varying_long", "mixed")]
INT8_ON_COMPILE = [1 - c3[k] / c2[k] for k in ("varying_short", "varying_medium", "varying_long", "mixed")]


def preds(run):
    return pd.read_parquet(run["pred"])[["text_hash", "p_unsafe"]]


subset = set(preds(COMPILE[(HTG, "none", True)]).text_hash)
pairs = [("fp16 eager → fp16 + compile", EAGER[(HTG, "none", False)], COMPILE[(HTG, "none", True)]),
         ("fp16 eager → INT8 torchao eager", EAGER[(HTG, "none", False)], EAGER[(HTG, "torchao_int8_dyn", False)]),
         ("INT8 torchao eager → INT8 torchao + compile", EAGER[(HTG, "torchao_int8_dyn", False)], COMPILE[(HTG, "torchao_int8_dyn", True)]),
         ("fp16 + compile → INT8 torchao + compile", COMPILE[(HTG, "none", True)], COMPILE[(HTG, "torchao_int8_dyn", True)])]
flip_rows = []
for name, a, b in pairs:
    j = preds(a)[lambda d: d.text_hash.isin(subset)].merge(preds(b), on="text_hash", suffixes=("_a", "_b"))
    f = int(((j.p_unsafe_a > .5) != (j.p_unsafe_b > .5)).sum())
    dp = (j.p_unsafe_a - j.p_unsafe_b).abs()
    flip_rows.append({"переход": name, "N": len(j), "смен": f"{f} ({100 * f / len(j):.1f}%)",
                      "средний \\|Δp\\|": f"{dp.mean():.4f}", "макс \\|Δp\\|": f"{dp.max():.3f}"})
compile_flip_md = md(pd.DataFrame(flip_rows))
ct = read(rep(R_COMPILE, HTG, "torchao_int8_dyn"), "tail")
compile_tail_md = md(pd.DataFrame([{"хвост": r.tail, "N": int(r.n), "смен, %": pct(r.flip_rate),
                                    "хуже / лучше": f"{int(r.worse)} / {int(r.better)}",
                                    "ΔFNR, п.п.": pct(r.delta_fnr, sign=True), "ΔFPR, п.п.": pct(r.delta_fpr, sign=True)}
                                   for r in ct.itertuples()]))
co = read(rep(R_COMPILE, HTG, "torchao_int8_dyn"), "overall").set_index("aggregate").loc["pooled"]

# ---------------------------------------------------------------- профиль
layers = pd.read_csv(R_PROFILE / "layers.csv").rename(columns={
    "gpu_kernel_ms": "ядра GPU, мс", "kernels": "ядер", "syncs": "синхронизаций", "memcpy": "копирований", "cpu_ops": "CPU-операций"})
layers_md = md(layers)
models_prof_md = md(pd.read_csv(R_PROFILE / "models.csv"))

# ---------------------------------------------------------------- среда
env = EAGER[(HTG, "none", False)]["manifest"]["environment"]
pk = env["packages"]
ENV = (f"Kaggle, Tesla T4 ({env.get('capability')}, {env.get('gpu_memory_gb')} ГБ), драйвер {env.get('driver')}, CUDA {env.get('cuda_runtime')}, "
       f"Python {env['python']}, torch {pk['torch']}, transformers {pk['transformers']}, torchao {pk['torchao']}, "
       f"bitsandbytes {pk['bitsandbytes']}, gliner2 {pk['gliner2']}")

# ---------------------------------------------------------------- текст
per_model = []
for model in MODELS:
    note = ""
    if model == "koala":
        note = ("\n> KoalaAI — англоязычная модель модерации контента: на этом наборе почти всё считает безопасным "
                "(FNR эталона 97–100%). Смен вердиктов почти нет, потому что менять нечего; в выводах о квантовании не учитывается.\n")
    per_model.append(f"""### {TITLE[model]} — {KIND[model]}
{note}
**По сплитам.** Ошибка сплита — FNR для сплитов с вредными текстами, FPR для безопасных. «Смен» — доля изменённых вердиктов
относительно fp16, в скобках сколько стали неверными (↓) и сколько исправились (↑). ⚠ — ошибка выросла на 1 п.п. и больше.

{split_table(model)}
**По хвостам.**

{tail_table(model)}
**Скорость по бакетам** (батч 1, полный запрос «текст → вердикт», p50 / p95 в мс; Δp95 — относительно fp16).{TOKNOTE[model]}

{speed_table(model)}""")

f = lambda xs: f"{100 * min(xs):.0f}–{100 * max(xs):.0f}%"
report = f"""# Динамическое квантование guardrail-моделей на Kaggle T4: качество на хвостах и скорость

Этап 2 проекта AutoTune Guardrails (роль У2: датасет хвостов и baseline динамического квантования). Все цифры — из прогонов
инструмента `guard_eval`; среда: {ENV}.

## 1. Что решили проверить

Исходный запрос кейсодержателя — **динамическое квантование** guardrail-моделей: веса квантуются заранее, активации — на лету,
без калибровки. Целевая метрика качества — **число изменённых вердиктов** относительно неквантованной модели; метрики считаются
по каждому сплиту, по бенчмаркам и в целом. Цель проекта из презентации: ≥20% снижения p95 и **0 деградации на хвостах**
(неанглийские тексты, код, обфускация).

Гипотезы перед экспериментом:

| # | Гипотеза | Откуда |
|---|---|---|
| H1 | Динамическое INT8 ускоряет инференс guardrail-моделей | заказ кейсодержателя |
| H2 | Квантование сильнее ухудшает хвосты, чем среднее качество | LLM.int8(), SmoothQuant: выбросы в активациях; Marie & Fujita (2025): редкие языки |
| H3 | LLM.int8() (bitsandbytes, выбросы в fp16) сохраняет качество лучше, чем простое W8A8 (torchao) | LLM.int8() |
| H4 | Выигрыш от INT8 больше у крупных decoder-моделей на длинных промптах, чем у маленьких encoder | профиль этапа 1: маленькие модели упираются в overhead |
| H5 | FP8 — лучший компромисс скорости и качества | диапазон FP8 шире, чем у INT8 |

## 2. Что сделали

**Датасет хвостов `tails_v1`** — приватный `AutotuneGuardrails/tails`, {N_TOTAL} строк:

{ds_md}
RTP-LX: токсичность промпта ≥ 4 — harm, ≤ 2 — safe, 3 исключено. В `hivetrace_input_robust` 8 аугментаций HiveTrace;
`informal_rewrited` отнесена к `paraphrase`, остальные — к `obfuscation`. Ответы LLM (`robustness-test-output`), синтетика
(код, кодировки, be/uz) и закрытые наборы кейсодержателя в v1 не вошли.

**Инструмент `guard_eval`:** любая модель через spec (YAML), три метода динамического квантования `nn.Linear`
(головы классификаторов и `lm_head` не квантуются), кэш предсказаний по хешу конфигурации и текста, метрики по сплитам,
бенчмаркам, хвостам и в целом (pooled и macro по бенчмаркам) с bootstrap-интервалами, сравнение двух конфигураций
(смены вердиктов, хуже/лучше, McNemar, ΔFNR/ΔFPR/Δmacro-F1), замер скорости по бакетам `synthetic-v1` из ветки У1.

**Модели и конфигурации:**

| Модель | Тип | Вердикт |
|---|---|---|
| gliner-guard-omni | encoder 307M | логиты классификатора safety |
| KoalaAI/Text-Moderation | encoder 139M | 1 − P(OK) |
| HiveTraceGuard-Pro | decoder 0.6B | логиты `safe`/`unsafe` первого токена, шаблон с политикой (~420 токенов) |
| OpenGuardrails-Text-4B-0124 | decoder 4B | то же, шаблон версии 2510 (инструкция ~290 токенов) |

Методы: эталон fp16 (bf16 у T4 нет); `bnb_int8` — LLM.int8(); `torchao_int8_dyn` — W8A8 dynamic; `torchao_fp8_dyn` —
**не поддерживается на T4** (нужен sm_89+). Порог вердикта 0.5. Все прогоны — без ошибок.

## 3. Результаты

### 3.1. Сводка: квантованные версии относительно fp16 (eager, полный набор)

{overall_md}
«Хуже / лучше» — сколько вердиктов стали неверными и сколько исправились. p95 mixed — бакет со смешанными длинами.

### 3.2. Качество эталонов fp16 по бенчмаркам

{bench_md}
HiveTraceGuard-Pro — лучший на продуктовом домене HiveTrace; gliner-guard лучше всех ловит редкие языки (MultiJail, CSRT),
но даёт больше ложных срабатываний; у всех моделей, кроме KoalaAI, FPR на безопасной части RTP-LX около 35% — стоит проверить порог разметки (safe = токсичность 1–2).

### 3.3. По моделям: сплиты, хвосты, скорость

{chr(10).join(per_model)}
### 3.4. Обфускация: смены вердиктов по аугментациям `hivetrace_input_robust`, %

{aug_md}
### 3.5. Где квантование бьёт сильнее всего: три сплита с наибольшим ростом ошибки

{worst_md}
Сильнее всего растёт ошибка на редких языках MultiJail (бенгальский, итальянский, вьетнамский); среди аугментаций
(раздел 3.4) больше всего смен даёт неправильная раскладка `wrong_layout_ru_to_en`. То есть страдают именно хвосты,
ради которых проект и проверяет оптимизации.

## 4. Дополнительные эксперименты

### 4.1. Профилирование: откуда замедление квантованных моделей

Квантованные версии в eager оказались медленнее fp16, причём с постоянной добавкой на каждый квантованный слой
(~2 мс у torchao, ~0.2–0.7 мс у bnb), не зависящей от длины входа. Профиль одного слоя `nn.Linear` и целой модели:

**Один слой** (вызов в мс; число ядер, синхронизаций CPU↔GPU, копирований и CPU-операций на вызов):

{layers_md}
**Модель, один запрос** (~128 токенов текста, батч 1):

{models_prof_md}
Выводы профиля:
- **torchao в eager:** `quant::choose_qparams_affine` (масштаб активаций) вызывает `aten::_local_scalar_dense` — около 48
  синхронизаций и копирований GPU→CPU на каждый слой. Само INT8-умножение (`aten::_int_mm`) на T4 работает на GPU.
  У HiveTraceGuard это 9 413 синхронизаций на запрос, GPU занята 41%.
- **с `torch.compile`** синхронизации исчезают (0 на слой, 3 ядра вместо 32), слой ускоряется в ~8.5 раза.
- **bnb** считает на INT8 Tensor Cores (`cutlass_75_tensorop_i8816gemm`), но поиск выбросов даёт 2 синхронизации на слой.
- **KoalaAI** упирается в CPU (GPU занята 32% даже в fp16): compile не помогает, нужны CUDA Graphs.

### 4.2. `torch.compile`: честное сравнение compile против compile (HiveTraceGuard-Pro)

Скорость, p50 / p95 в мс (в скобках — изменение p95 относительно fp16 eager):

{compile_speed_md}
- **Компиляция сама по себе** снижает p95 на {f(COMPILE_GAIN)}.
- **INT8 поверх компиляции** добавляет ещё {f(INT8_ON_COMPILE)} по p95 (больше на длинных входах) и экономит ~30% памяти.

Смены вердиктов на подвыборке (50 текстов на сплит, {len(subset)} текстов; в ней перевес хвостов — 500 текстов MultiJail):

{compile_flip_md}
fp16 + compile → INT8 + compile (pooled): {int(co.flips)} смен ({pct(co.flip_rate)}% [{pct(co.flip_rate_lo)}; {pct(co.flip_rate_hi)}]),
хуже / лучше {int(co.worse)} / {int(co.better)}, ΔFNR {pct(co.delta_fnr, sign=True)} п.п., ΔFPR {pct(co.delta_fpr, sign=True)} п.п.
По хвостам:

{compile_tail_md}
## 5. Выводы по гипотезам

| # | Гипотеза | Итог |
|---|---|---|
| H1 | INT8 ускоряет | **Нет в eager:** все 8 конфигураций медленнее fp16 (от +2% у OpenGuardrails/bnb на длинных входах до ×9 по p50 у gliner/torchao). **С compile — немного:** у HiveTraceGuard INT8 даёт {f(INT8_ON_COMPILE)} поверх fp16 + compile. Пиковая память — {MEM_RANGE} |
| H2 | Хвосты страдают сильнее | **Да:** на `in_domain` меняется 0.4–1.0% вердиктов, на неанглийских текстах и обфускации — до 2.9–5.8% |
| H3 | LLM.int8 бережнее, чем W8A8 | **Да:** HiveTraceGuard 2.1% против 4.0% смен (FNR +0.2 против +2.6 п.п.), OpenGuardrails 2.0% против 2.6%, gliner 1.1% против 1.5% |
| H4 | Выигрыш больше у крупных decoder | **Частично:** замедление наименьшее у OpenGuardrails (bnb на длинных входах +2% по p95), у INT8 + compile выигрыш растёт с длиной входа |
| H5 | FP8 | **Не проверено:** нет поддержки на T4 |
| Этап 1 | Ускорение без квантования | **Да:** `torch.compile` даёт {f(COMPILE_GAIN)} по p95 при 0.1% смен вердиктов |

**Главное:** цель ≥20% по p95 достигается lossless-оптимизацией (`torch.compile`), а динамическое INT8 на T4 добавляет немного
скорости ценой смен вердиктов на хвостах — у HiveTraceGuard в основном к худшему (модель чаще пропускает вредные запросы).
Требование «0 деградации на хвостах» ни один INT8-вариант не выполняет.

## 6. Ограничения

- T4 — не продуктовое железо (продукт работает на Blackwell); FP8 не проверен.
- Скорость eager и compile снята в разных сессиях Kaggle и без парного A/B из протокола У1; расхождение между сессиями
  наблюдалось около ±7%, эффект compile (×2) его заметно превышает.
- Качество с compile — на подвыборке {len(subset)} текстов (широкий интервал), только HiveTraceGuard.
- Вес модели для torchao в `SUMMARY.md` посчитан неверно (тензоры torchao не учитываются) — экономию памяти смотреть по пиковой памяти.
- Один прогон на конфигурацию; порог 0.5 для всех моделей; KoalaAI на этом наборе неинформативна.

## 7. Переход на RTX 5090 и гипотезы

Следующие прогоны переводятся на **RunPod с RTX 5090** (Blackwell, sm_120, 32 ГБ): ближе к продуктовому железу. Команды
`guard_eval` те же; `dtype: auto` выберет bf16.

Гипотезы:

| # | Гипотеза для 5090 | Как проверим |
|---|---|---|
| G1 | **Компиляция останется главным источником ускорения.** GPU намного быстрее T4, поэтому на батче 1 модели ещё сильнее упрутся в CPU и накладные расходы; для маленьких моделей (KoalaAI, gliner, HiveTraceGuard на коротких входах) понадобятся CUDA Graphs (`mode="reduce-overhead"`) | bf16 eager / compile / compile + CUDA Graphs по бакетам |
| G2 | **INT8 в eager останется медленным:** синхронизации в torchao — свойство реализации, а не GPU | профиль одного слоя на 5090 |
| G3 | **FP8 dynamic + compile даст лучший баланс:** аппаратный FP8 у Blackwell и более широкий диапазон, чем у INT8, — меньше смен вердиктов на хвостах при сопоставимом ускорении | FP8 против INT8 против bf16, все с compile |
| G4 | **Смены вердиктов от метода почти не зависят от GPU:** у INT8 ожидаем те же 1–6% на хвостах; эталон сменится с fp16 на bf16, поэтому цифры пересчитаем заново | полный набор, compile для всех вариантов |
| G5 | **Выигрыш от квантования будет заметен только у крупных моделей на длинных входах** (OpenGuardrails 4B, WildGuard 7B), у маленьких — около нуля | те же бакеты, плюс WildGuard (на T4 эталон не помещался) |
| G6 | **bnb останется медленнее bf16:** синхронизации заложены в алгоритм LLM.int8() | бакеты скорости |

План прогонов на 5090:
1. `python -m guard_eval check` — поддержка трёх методов и FP8 на sm_120.
2. Все модели + WildGuard: bf16 eager и bf16 + compile — эталоны скорости и качества.
3. `torchao_int8_dyn`, `torchao_fp8_dyn` с compile и `bnb_int8` — полный набор, смены вердиктов по сплитам и хвостам.
4. Скорость по парному протоколу У1 (A/A и A/B) для финалистов; CUDA Graphs для encoder.
5. Профиль одного слоя и модели на 5090 для каждого метода.

## 8. Данные и воспроизведение

- Датасет: `AutotuneGuardrails/tails` (приватный), сборка — `stage2/build_tails.py`.
- Инструмент `guard_eval/` и ноутбук Kaggle `notebooks/guard_eval_kaggle.ipynb`; как запускать — `docs/GUARD_EVAL.md`.
- Прогоны (скоры по `text_hash`, манифесты, `speed.json`): `results/t4/eager/runs/`,
  профиль — `results/t4/profile/`, compile — `results/t4/compile/`.
- Тексты изменённых вердиктов (`flips.csv`, секции в `summary.md`) в репозиторий не выложены: в них тексты из закрытых наборов.
"""
OUT.write_text(report)
print(OUT, len(report.splitlines()), "строк")
