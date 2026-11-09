# Сводка сравнений

| модель | A | B | N | смен | хуже/лучше | macro смен | macro ΔFNR | macro ΔFPR | веса A→B, МБ | p95 ↓ varying_short | p95 ↓ varying_medium | p95 ↓ varying_long | p95 ↓ mixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hivetrace-guard-pro | hivetrace-guard-pro__none__float16 | hivetrace-guard-pro__torchao_int8_dyn__float16 | 950 | 53 (5.6% [4.1%; 7.1%]) | 43/10 | 3.4% | 2.3% | -1.3% | 1137→1137 | 6.1% | 12.4% | 15.7% | 15.9% |

смен — изменённые вердикты B относительно A (pooled, 95% bootstrap CI); p95 ↓ — снижение p95 полного запроса.
