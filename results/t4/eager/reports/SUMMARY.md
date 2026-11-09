# Сводка сравнений

| модель | A | B | N | смен | хуже/лучше | macro смен | macro ΔFNR | macro ΔFPR | веса A→B, МБ | p95 ↓ varying_short | p95 ↓ varying_medium | p95 ↓ varying_long | p95 ↓ mixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gliner-guard | gliner-guard__none__float16 | gliner-guard__bnb_int8__float16 | 14249 | 150 (1.1% [0.9%; 1.2%]) | 78/72 | 1.0% | 0.5% | -0.9% | 586→481 | -242.2% | -248.0% | -231.3% | -245.0% |
| gliner-guard | gliner-guard__none__float16 | gliner-guard__torchao_int8_dyn__float16 | 14249 | 215 (1.5% [1.3%; 1.7%]) | 106/109 | 1.2% | 0.4% | -0.7% | 586→586 | -744.6% | -762.5% | -690.2% | -692.1% |
| hivetrace-guard-pro | hivetrace-guard-pro__none__float16 | hivetrace-guard-pro__bnb_int8__float16 | 14249 | 297 (2.1% [1.9%; 2.3%]) | 163/134 | 2.2% | -0.0% | 0.0% | 1137→717 | -155.7% | -113.7% | -57.0% | -56.2% |
| hivetrace-guard-pro | hivetrace-guard-pro__none__float16 | hivetrace-guard-pro__torchao_int8_dyn__float16 | 14249 | 564 (4.0% [3.7%; 4.3%]) | 376/188 | 4.0% | 2.6% | -0.2% | 1137→1137 | -388.8% | -293.5% | -155.1% | -154.1% |
| koala | koala__none__float16 | koala__bnb_int8__float16 | 14249 | 6 (0.0% [0.0%; 0.1%]) | 6/0 | 0.0% | 0.0% | 0.0% | 266→171 | -182.1% | -172.6% | -174.0% | -189.3% |
| koala | koala__none__float16 | koala__torchao_int8_dyn__float16 | 14249 | 10 (0.1% [0.0%; 0.1%]) | 8/2 | 0.1% | 0.0% | 0.0% | 266→266 | -562.8% | -534.0% | -511.1% | -525.9% |
| openguardrails | openguardrails__none__float16 | openguardrails__bnb_int8__float16 | 14249 | 278 (2.0% [1.7%; 2.2%]) | 120/158 | 2.2% | -0.9% | 0.3% | 7672→4207 | -25.1% | -19.2% | -1.9% | -2.4% |
| openguardrails | openguardrails__none__float16 | openguardrails__torchao_int8_dyn__float16 | 14249 | 374 (2.6% [2.4%; 2.9%]) | 168/206 | 3.0% | -1.6% | 1.1% | 7672→7672 | -193.7% | -167.4% | -98.4% | -99.3% |

смен — изменённые вердикты B относительно A (pooled, 95% bootstrap CI); p95 ↓ — снижение p95 полного запроса.
