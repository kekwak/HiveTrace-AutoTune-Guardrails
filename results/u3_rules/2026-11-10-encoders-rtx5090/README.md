# Энкодеры, RTX 5090: последовательное добавление правил

Прогон 10–11 ноября 2026 на поде RunPod `wi79w6b6bwhezg` (Secure, EU-RO-1): RTX 5090, драйвер 595.91.07,
AMD Ryzen 9 9950X (32 потока), 123 GB RAM, torch 2.11.0+cu130. Измеритель У1 — коммит `2b09a3e`,
режим `screening`, batch 1. Датасет `role-corpus`, manifest `760adad3…`.

На каждом шаге добавляется одно правило к предыдущим; baseline меряется заново в каждом прогоне.
Порядок: llama `u1→c8→u4→c12→u3`, laya `u1→c9→u4`, koala `u1→c8→u4→c12→u3`, gliner `u1→c8→u4→c12`.

В `raw/` есть ещё контрольный прогон laya без u1 (`laya-c9`, `c9→u4→u3`); в таблицы он не вошёл.

## Файлы

| Файл | Что внутри |
|---|---|
| `RESULTS.md` | таблицы по моделям: p95/p50, Δp95, A/A, изменённые вердикты, статус бакета и прогона |
| `results.csv` | то же в CSV, одна строка на модель × шаг × бакет |
| `results_full.csv` | все поля `summary.json`: интервалы ratio, блоки, дрейф, причины, id изменённых вердиктов, макс. Δscore |
| `memory.csv` | память по процессам: 3 baseline + 3 candidate на шаг, пики GPU/RSS, счётчики torch |
| `raw/` | весь `runs/tune` с пода без изменений (6064 файла, сверены sha256), плюс `machine.txt` |

Таблицы в корне собраны без `laya-c9`. `raw/RESULTS.md` и `raw/results.csv` пересобраны на Mac после прогона (`rules/table.py raw`), чтобы показать статус прогона;
сами замеры в `raw/` не менялись.

## Как воспроизвести

```bash
rules/pod_sync.sh root@<ip> <port>                    # с Mac: rules/ на под
ssh -p <port> root@<ip> /workspace/guardrails/rules/pod_setup.sh   # код, окружение, датасет, веса
ssh -p <port> root@<ip> /workspace/guardrails/rules/pod_run.sh     # прогоны + RESULTS.md/results.csv
python3 rules/export.py <runs/tune> <каталог>         # results_full.csv и memory.csv
```
