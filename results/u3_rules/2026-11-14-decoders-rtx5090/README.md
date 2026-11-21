# Декодеры, RTX 5090: важные правила, batch 1

Прогон 14 ноября 2026 на поде RunPod `55nwtcn1mmmc2j` (Secure, EUR-NO-1): RTX 5090, драйвер 595.91.07,
AMD EPYC 7543, torch 2.11.0+cu130. Измеритель У1 v4 — коммит `100ce48` с патчем `raw/measurer.patch`
(см. ниже), batch 1, бакеты B1–B4. Индекс датасета собран `prepare-data` по `configs/v4/reindex.json`
для HaloGuard, HiveTraceGuard-Pro и ShieldGemma, manifest `b5c22d20…` (`raw/data.json`).

Шаг 0 — baseline, шаги 1+ — режим `search` по нему: только candidate, без A/A и парных интервалов,
статус `exploratory`. Проверяли только важные правила, U1+U4 одним шагом:
HaloGuard `C2 → +U1+U4 → +C7`, HiveTraceGuard-Pro `C13 → +C2 → +U1+U4`, ShieldGemma `U1+U4 → +C3`.
OpenGuardrails отложен, см. `2026-11-14-open-rtx4090`.

## Итог (p95, мс; вердикты из 400)

| Модель | Набор | B1 | B2 | B3 | B4 | Вердикты |
|---|---|---|---|---|---|---|
| HaloGuard | C2+U1+U4+C7 | 62.2 → 8.0 (−87%) | 57.3 → 19.7 (−66%) | 62.2 → 36.1 (−42%) | 97.9 → 68.1 (−30%) | 0 |
| ShieldGemma | U1+U4+C3 | 51.8 → 14.0 (−73%) | 54.3 → 26.2 (−52%) | 83.6 → 58.0 (−31%) | 188.5 → 131.0 (−31%) | 1 / 0 / 0 / 1 |
| ShieldGemma | U1+U4 | 51.8 → 14.6 (−72%) | 54.3 → 27.3 (−50%) | 83.6 → 57.4 (−31%) | 188.5 → 133.9 (−29%) | 1 / 0 / 0 / 1 |
| HiveTraceGuard-Pro | C13+C2+U1+U4 | 47.4 → 42.5 (−10%) | 46.6 → 42.8 (−8%) | 58.5 → 56.6 (−3%) | 57.5 → 101.6 (+77%) | 0 / 1 / 3 / 0 |

- HaloGuard и ShieldGemma: снижение p95 больше 20% во всех бакетах. У HaloGuard скоры сдвинулись максимум на 0.03, вердикты те же; у ShieldGemma по одному пограничному вердикту в B1 и B4.
- U1+U4 даёт основной выигрыш на коротких входах; C7 вытягивает длинные бакеты HaloGuard (без него B4 был +17%).
- C3 у ShieldGemma добавляет 1–2%.
- C2 (кэш префикса) в baseline v4 вредит: HaloGuard +10…+18% p95 на B1–B3, HiveTraceGuard-Pro до +72% на B4 и 4 изменённых вердикта.
  Поэтому итог HiveTraceGuard-Pro искажён C2; вариант C13+U1+U4 без C2 измерен 21 ноября (`2026-11-21-hive-rtx5090`).
- C13 у HiveTraceGuard-Pro сам по себе почти ничего не даёт: baseline v4 уже генерирует один токен.

## Патч измерителя

`u1_benchmark/v4_privacy.py`: публичная копия конфига прячет всё под ключом `prompt`, включая ссылку
`sdk_assets.prompt` у HaloGuard (`repo`/`revision`), после чего валидатор отчёта отвергает её
(`invalid_frozen_source`) — отчёт HaloGuard не собирается даже у чистого baseline. Патч оставляет
ссылки `{repo, revision, ...}` открытыми, текст промпта по-прежнему скрыт. На замеры не влияет.

## Файлы

| Файл | Что внутри |
|---|---|
| `RESULTS.md` | таблицы по моделям: p95/p50, Δp95, изменённые вердикты, статус |
| `results.csv` | то же в CSV, одна строка на модель × шаг × бакет |
| `results_full.csv` | все скалярные поля ячеек `summary.json` v4 |
| `memory.csv` | `worker_evidence`: память и время подготовки по каждому процессу |
| `raw/` | весь `runs/tune-v4` с пода (2894 файла, сверены sha256), `machine.txt`, `data.json`, `measurer.patch` |

## Как воспроизвести

```bash
rules/pod_sync.sh root@<ip> <port>
ssh -p <port> root@<ip> 'read -r HF_TOKEN; export HF_TOKEN; /workspace/guardrails/rules/pod_setup.sh halo hive shield'   # токен через stdin
# применить raw/measurer.patch в /workspace/guardrails
ssh -p <port> root@<ip> /workspace/guardrails/rules/pod_run.sh 1 halo hive shield
python3 rules/export.py <runs/tune-v4> <каталог>
```
