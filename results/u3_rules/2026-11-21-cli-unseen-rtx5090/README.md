# Переносимость правил на две новые модели, RTX 5090, batch 1

Прогон 21 ноября 2026 на поде RunPod `55nwtcn1mmmc2j` (Secure, EUR-NO-1): RTX 5090, драйвер 595.112.07.
Обе модели раньше не запускались. Правила выбирал `guardrail-tuner plan` по `config.json` (CLI `e146df0`),
руками ничего не настраивалось. Тексты — корпус `role-v4` (`vXofi/guardrails-speed-en-ru`, ревизия `7324b88`),
тот же, что у остальных моделей: по 400 текстов на бакет, бакеты по числу токенов модели.

Шаг 0 — без правил, шаг 1 — правила из плана. Один проход после прогрева, без A/A и парных интервалов.

## Итог (p95, мс; вердикты из 400)

| Модель | Правила из плана | B1 | B2 | B3 | B4 | Вердикты |
|---|---|---|---|---|---|---|
| protectai DeBERTa v3 | C12, C8, U4, U1 | 19.6 → 6.1 (−69%) | — | — | — | 0 |
| AccuKnox Guard 1.1 | C7, C12 | 92.7 → 57.6 (−38%) | 136.4 → 56.7 (−58%) | 221.0 → 109.1 (−51%) | 353.8 → 197.3 (−44%) | 0 |

- **protectai** замерен командами CLI: `plan` → `serve` с `rules: []` и с правилами из плана, запросы по одному тексту.
  Время включает HTTP на localhost и токенизацию. В конфиге из плана изменены `max_length` 128 → 512 и
  `batch_size` 4 → 1, как в замерах остальных моделей. `run --config` на тех же текстах: все четыре правила
  применены, сверка с eager пройдена, максимальная разница логитов 6.7e-6.
- **AccuKnox** CLI только планирует: `execution` пустой, `run` и `serve` её не запускают. Замер сделан скриптом
  `rules/accuknox_check.py`: он применяет правила тем же модулем (`RuledModel(model.backbone, rules, None)`)
  и меряет `model.system_one(...)`. Шаг 0 — без пакета `flash-linear-attention`, шаг 1 — с ним: его требует C7.
- AccuKnox считает безопасными все 1600 текстов, поэтому сверка вердиктов слабая. Максимальный сдвиг скора 0.0035.
- U1 и U4 для AccuKnox план не включает: без контракта вызова CLI не знает, какой forward оборачивать.
- Бакеты AccuKnox посчитаны вместе с её служебным промптом в 143 токена (`raw/accuknox-prompt.json`).

## Условия

- **CLI:** ветка `feat/guardrail-tuner-cli`, torch 2.15.0+cu132, transformers 4.57.6.
- **Окружение AccuKnox:** torch 2.13.0+cu130, transformers 5.17.0, triton 3.7.1, `flash-linear-attention` 0.5.2.
  Ядро `causal_conv1d` не ставилось ни в одном шаге.
- **Модели:** protectai `90c9989`; AccuKnox `e4e43d8`, bf16.

## Файлы

| Файл | Что внутри |
|---|---|
| `RESULTS.md` | таблицы по моделям: p95/p50, Δp95, изменённые вердикты |
| `results.csv` | то же в CSV, одна строка на модель × шаг × бакет |
| `raw/` | планы CLI, конфиги `serve`, результаты по бакетам (`*.jsonl`), логи, `machine.txt`, sha256 наборов текстов |

## Как воспроизвести

```bash
python rules/cli_texts.py protectai/deberta-v3-base-prompt-injection-v2 512 > texts-protectai.json
python rules/cli_check.py texts-protectai.json protectai/deberta-v3-base-prompt-injection-v2 512 > raw/protectai.jsonl
python rules/cli_texts.py Accuknoxtechnologies/accuknox-guard-1.1 4096 143 > texts-accuknox.json
python rules/accuknox_check.py texts-accuknox.json > raw/accuknox.jsonl          # без flash-linear-attention
python rules/accuknox_check.py texts-accuknox.json c7,c12 >> raw/accuknox.jsonl  # с flash-linear-attention
python rules/cli_table.py .
```
