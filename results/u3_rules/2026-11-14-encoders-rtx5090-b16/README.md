# Энкодеры, RTX 5090, batch 16: последовательное добавление правил

Прогон 14 ноября 2026 на поде RunPod `wi79w6b6bwhezg` (Secure, EU-RO-1): RTX 5090, драйвер 595.91.07,
AMD Ryzen 9 9950X (32 потока), 123 GB RAM, torch 2.11.0+cu130. Измеритель У1 v4 — коммит `100ce48`,
batch 16, бакет B1 (до 512 токенов). Индекс датасета собран `prepare-data` по `configs/v4/reindex.json`
только для четырёх энкодеров, manifest `7ea1621b…` (`raw/data.json`).

Шаг 0 — baseline (режим `baseline`), шаги 1+ — режим `search` по этому baseline: меряется только candidate,
без A/A и без парных доверительных интервалов, статус `exploratory`. Порядок правил тот же, что для batch 1:
llama `u1→c8→u4→c12→u3`, laya `u1→c9→u4`, koala `u1→c8→u4→c12→u3`, gliner `u1→c8→u4→c12`.
U3 в батче не применяется: маска нужна для паддинга.

p95/p50 — время одного батча из 16 текстов, мс.

## Итог

| Модель | Лучший набор | p95 батча, мс | Ускорение | Изменённые вердикты |
|---|---|---|---|---|
| Llama Prompt Guard 2 | U1+C8+U4+C12 | 41.5 → 24.9 | −40% | 0 / 400 |
| GLiNER Guard | U1+C8+U4+C12 | 93.1 → 57.1 | −39% | 0 / 400 |
| Laya | U1+C9+U4 | 64.3 → 47.0 | −27% | 0 / 400 |
| KoalaAI | U1+C8+U4 | 97.9 → 74.9 | −23% | 0 / 400 |

- Выигрыш при batch 16 почти весь даёт U4 (`torch.compile`). U1 (CUDA graph) и C8 сами по себе дают от −3% до +5%:
  при батче накладные расходы CPU на запуск ядер уже не главная доля времени.
- У Laya на шагах U1 и U1+C9 один пограничный вердикт из 400 изменился, на финальном шаге — ноль.
- Цифры batch 1 из `2026-11-10-encoders-rtx5090` сняты старым измерителем (`2b09a3e`) с другим baseline, напрямую их не сравнивать.

## Файлы

| Файл | Что внутри |
|---|---|
| `RESULTS.md` | таблицы по моделям: p95/p50, Δp95, изменённые вердикты, статус |
| `results.csv` | то же в CSV, одна строка на модель × шаг |
| `results_full.csv` | все скалярные поля ячеек `summary.json` v4 |
| `memory.csv` | `worker_evidence` из `summary.json`: память, время подготовки по каждому процессу |
| `raw/` | весь `runs/tune-v4-b16` с пода без изменений (2836 файлов, сверены sha256), `machine.txt`, `data.json` |

## Как воспроизвести

```bash
rules/pod_sync.sh root@<ip> <port>
ssh -p <port> root@<ip> /workspace/guardrails/rules/pod_setup.sh
ssh -p <port> root@<ip> /workspace/guardrails/rules/pod_run.sh 16
python3 rules/export.py <runs/tune-v4-b16> <каталог>
```
