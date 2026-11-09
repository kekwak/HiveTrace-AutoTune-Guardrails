# HiveTrace-AutoTune-Guardrails — research/quantization

Исследовательская ветка: динамическое квантование guardrail-моделей на Kaggle T4 — смены вердиктов на хвостах и скорость.
Код CLI живёт в `main`; здесь только инструмент замеров `guard_eval`, ноутбук прогонов и полученные артефакты.
Полный отчёт — [`docs/report_quantization_T4.md`](docs/report_quantization_T4.md).

## Постановка

- **Модели:** gliner-guard-omni (encoder 307M), KoalaAI/Text-Moderation (encoder 139M), HiveTraceGuard-Pro (decoder 0.6B),
  OpenGuardrails-Text-4B-0124 (decoder 4B, шаблон версии 2510).
- **Методы:** эталон fp16; `bnb_int8` — LLM.int8(); `torchao_int8_dyn` — W8A8 dynamic. `torchao_fp8_dyn` на T4 не
  поддерживается (нужен sm_89+). Квантуются `nn.Linear`, головы и `lm_head` — нет. Порог вердикта 0.5.
- **Данные:** `AutotuneGuardrails/tails` (приватный HF), 14 249 строк: HiveTrace input / robust (ru, обфускация),
  RTP-LX (ru, uk), MultiJail (10 языков), CSRT.
- **Метрика качества:** число изменённых вердиктов относительно fp16 (по сплитам, бенчмаркам, хвостам; bootstrap CI, McNemar).
- **Скорость:** батч 1, полный запрос «текст → вердикт», бакеты `synthetic-v1`.
- **Среда:** Kaggle, Tesla T4, CUDA 12.8, torch 2.11.0+cu128, transformers 5.20.1, torchao 0.10.0, bitsandbytes 0.50.2.

## Результаты (eager, полный набор)

| модель | метод | смен вердиктов | хуже / лучше | macro ΔFNR, п.п. | p95 mixed, мс | пик памяти, МБ |
|---|---|---|---|---|---|---|
| gliner-guard-omni | LLM.int8() | 150 (1.1%) | 78 / 72 | +0.5 | 27.6 → 95.1 | 646 → 538 |
| gliner-guard-omni | torchao W8A8 | 215 (1.5%) | 106 / 109 | +0.4 | 27.6 → 218.3 | 646 → 537 |
| KoalaAI/Text-Moderation | LLM.int8() | 6 (0.0%) | 6 / 0 | +0.0 | 27.2 → 78.7 | 328 → 225 |
| KoalaAI/Text-Moderation | torchao W8A8 | 10 (0.1%) | 8 / 2 | +0.0 | 27.2 → 170.2 | 328 → 230 |
| HiveTraceGuard-Pro | LLM.int8() | 297 (2.1%) | 163 / 134 | -0.0 | 187.9 → 293.5 | 1376 → 995 |
| HiveTraceGuard-Pro | torchao W8A8 | 564 (4.0%) | 376 / 188 | +2.6 | 187.9 → 477.5 | 1376 → 997 |
| OpenGuardrails-Text-4B | LLM.int8() | 278 (2.0%) | 120 / 158 | -0.9 | 577.4 → 591.2 | 8046 → 4560 |
| OpenGuardrails-Text-4B | torchao W8A8 | 374 (2.6%) | 168 / 206 | -1.6 | 577.4 → 1150.6 | 8046 → 4553 |

**`torch.compile` (HiveTraceGuard-Pro, p95 относительно fp16 eager):** fp16 + compile −51…58%; INT8 torchao + compile
−55…64% (то есть ещё 6–16% поверх fp16 + compile) и ~30% меньше памяти, но 5.6% смен вердиктов (43 хуже / 10 лучше,
ΔFNR +4.7 п.п.) на подвыборке 950 текстов.

**Профиль:** в eager torchao делает ~48 синхронизаций GPU→CPU на слой (`choose_qparams_affine`), bnb — 2; с compile
синхронизации исчезают. KoalaAI упирается в CPU даже в fp16.

## Выводы

- **H1, INT8 ускоряет — нет в eager:** все 8 конфигураций медленнее fp16. С compile — +6–16% у HiveTraceGuard. Память −17…43%.
- **H2, хвосты страдают сильнее — да:** `in_domain` 0.4–1.0% смен, неанглийские и обфускация — до 2.9–5.8%.
- **H3, LLM.int8() бережнее W8A8 — да:** HiveTraceGuard 2.1% против 4.0%, OpenGuardrails 2.0% против 2.6%.
- **H4, выигрыш больше у крупных decoder — частично.**
- **H5, FP8 — не проверено** (нет на T4).
- Цель ≥20% по p95 достигается lossless `torch.compile`; требование «0 деградации на хвостах» ни один INT8-вариант
  не выполняет. Дальше — RTX 5090 (bf16, FP8, CUDA Graphs), см. раздел 7 отчёта.

Ограничения: T4 — не продуктовое железо; eager и compile сняты в разных сессиях Kaggle (±7%); один прогон на конфигурацию.

## Структура

```
guard_eval/                 инструмент: spec → модель → квантование → предсказания, метрики, сравнения, скорость
specs/models/               spec четырёх моделей
scripts/                    quantize_model.sh, run_all.sh, run_all_2gpu.sh; make_report.py — сборка отчёта из results/t4
notebooks/                  guard_eval_kaggle.ipynb — прогоны на Kaggle 2×T4 и профилирование (с выводами)
docs/                       GUARD_EVAL.md — как запускать; report_quantization_T4.md — полный отчёт
results/t4/eager/           runs/ (скоры по text_hash, манифесты, speed.json), reports/ (SUMMARY.md, метрики), logs/
results/t4/compile/         HiveTraceGuard-Pro: fp16 + compile против INT8 + compile
results/t4/profile/         профиль одного слоя и модели: layers.csv, models.csv, таблицы операций *.txt
results/t4/tails_v1_stats.csv  состав датасета хвостов
```

Тексты изменённых вердиктов (`flips.csv` и соответствующие секции `summary.md`) и трейсы профайлера (*.json, ~1.6 ГБ)
в репозиторий не выложены. `scripts/make_report.py` перенаправлен на пути этой ветки и после переноса не запускался.
