# HiveTraceGuard-Pro без C2, RTX 5090, batch 1

Прогон 21 ноября 2026 на поде RunPod `55nwtcn1mmmc2j` (Secure, EUR-NO-1): RTX 5090, драйвер 595.112.07,
AMD EPYC 7543, torch 2.11.0+cu130. Измеритель У1 v4 — коммит `100ce48` с патчем `raw/measurer.patch`,
batch 1, бакеты B1–B4. Индекс датасета собран `prepare-data` по `configs/v4/reindex.json` только для
HiveTraceGuard-Pro, manifest `a3ba6ac4…` (`raw/data.json`). Входы те же, что 14 ноября: id текстов во всех
четырёх бакетах совпадают.

Шаг 0 — baseline, шаг 1 — режим `search` по нему: только candidate, без A/A и парных интервалов,
статус `exploratory`. Один шаг `C13+U1+U4`, без кэша префикса (C2).

## Итог (p95, мс; вердикты из 400)

| Набор | B1 | B2 | B3 | B4 | Вердикты |
|---|---|---|---|---|---|
| C13+U1+U4 | 45.0 → 6.8 (−85%) | 41.7 → 14.4 (−65%) | 45.8 → 32.9 (−28%) | 55.0 → 91.2 (+66%) | 0 / 2 / 5 / 0 |

- На входах до 2048 токенов (B1–B3) снижение p95 больше 20%. С C2 тот же набор давал −10 / −8 / −3%
  (`2026-11-14-decoders-rtx5090`): итог портил именно C2.
- B4 (до 4096 токенов) стал медленнее на 66%. Это совпадает с условием правила U1: для декодера без
  быстрых ядер linear attention граф на входах длиннее 2048 токенов включать нельзя. У HiveTraceGuard-Pro
  linear attention нет, адаптер измерителя это условие не проверяет.
- Изменились 7 вердиктов из 1600: 2 в B2 (unsafe → safe) и 5 в B3 (3 unsafe → safe, 2 safe → unsafe).
  Это в пределах допуска: до 5 из 400 считаем lossless. Для этой модели измеритель хранит только
  метку, без скора, поэтому сдвиг скора не измерен.
- C13 сам по себе 14 ноября вердикты не менял, значит их меняет U1+U4.

## Почему baseline новый

Сохранённый baseline от 14 ноября измеритель не принял: он сверяет драйвер, UUID GPU и хост, а под
после перезапуска попал на другую машину. Свежий baseline на тех же текстах: 45.0 / 41.7 / 45.8 / 55.0 мс
против 47.4 / 46.6 / 58.5 / 57.5 мс 14 ноября.

## Файлы

| Файл | Что внутри |
|---|---|
| `RESULTS.md` | таблица: p95/p50, Δp95, изменённые вердикты, статус |
| `results.csv` | то же в CSV, одна строка на шаг × бакет |
| `results_full.csv` | все скалярные поля ячеек `summary.json` v4 |
| `memory.csv` | `worker_evidence`: память и время подготовки по каждому процессу |
| `raw/` | весь `runs/tune-v4` с пода (546 файлов, сверены sha256), `machine.txt`, `data.json`, `measurer.patch` |

## Как воспроизвести

В `rules/drive.py` план для `hive` заменить на один шаг `["c13+u1+u4"]`.

```bash
rules/pod_sync.sh root@<ip> <port>
ssh -p <port> root@<ip> /workspace/guardrails/rules/pod_setup.sh hive
# применить raw/measurer.patch в /workspace/guardrails
ssh -p <port> root@<ip> /workspace/guardrails/rules/pod_run.sh 1 hive
python3 rules/export.py <runs/tune-v4> <каталог>
```
