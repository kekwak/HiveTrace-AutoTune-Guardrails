# convaiinnovations/laya — профилирование

**Выполнено в Colab 25.10.2026, одна Tesla T4.** PyTorch 2.11.0+cu130,
Transformers 4.57.1, 2 CPU-потока. Веса FP32 (1.57 GiB), SDK autocast FP16.

Ноутбук: [`notebooks/laya_profile.ipynb`](../../../notebooks/laya_profile.ipynb).

- Корневой English checkpoint, revision `7b928d828b7b0e022f929d9bd2e44165aa270148`.
- SDK `laya==0.3.20`, без Router, TileLang и compile; штатный mixed-precision forward.
- Один вопрос `choice` про опасные инструкции; `noul` не используется из-за оговорки model card.
- Batch=1; 5 прогревов, 50 замеров; forward отдельно от SDK predict e2e.
- Длины текста 16–1024 токена, но общий контекст — 512: длинные строки
  явно помечаются `truncated`, фактическая длина записывается в `input_tokens`.
- Smoke включает пустой текст, безопасный/опасный запрос, русский и обфускацию.
  Это не тест качества; корневая модель предназначена для английского.

## Запуск и артефакты
Загрузить notebook в Colab, выбрать GPU, выполнить сверху вниз в свежей session.
При необходимости добавить `HF_TOKEN` в Colab Secrets и разрешить notebook access.
Локальная переменная окружения не переносится в Colab автоматически.

Результаты в `/content/results/profiling/laya/<timestamp>/`:
`env.json`, `pip-freeze.txt`, `smoke.json`, `latency.csv`, `latency_samples.csv`,
`gpu_profile.csv`, `kernel_breakdown.csv`, `top_kernels_L*.csv`, `report.csv`,
`manifest.json`, `trace_L*.json`, `viz_*.json` и ZIP рядом с каталогом.
`manifest.json` со статусом completed создаётся только в конце.

Большие traces не включены в git. Локальный архив исходного прогона:
`docs/research/laya-20261025-071618.zip` (игнорируется); ноутбук создаёт их заново.

## Ключевые цифры

| Текст, токенов | Реальный вход | forward p50 / p95, мс | SDK e2e p50 / p95, мс |
|---|---|---|---|
| 16 | 50 | 29.91 / 35.88 | 32.88 / 41.30 |
| 128 | 162 | 29.52 / 34.18 | 32.61 / 36.05 |
| 256 | 290 | 31.70 / 35.54 | 35.44 / 40.69 |
| 512 | 512, усечён | 40.29 / 48.89 | 42.35 / 44.76 |
| 1024 | 512, усечён | 40.57 / 47.42 | 43.14 / 44.18 |

Инструкция и варианты добавляют 34 токена; для текста остаётся **478**.
L512/L1024 измеряют один и тот же сохранённый префикс, не длинный контекст.
Пик allocated памяти — **1.59–1.60 GiB**, без учёта CUDA context/allocator reserve.

## Наблюдаемые узкие места

1. **Повторные преобразования FP32→FP16.** На forward — 319 `aten::_to_copy`;
   только `float16_copy_kernel_cuda` занимает 9.14 мс при L16 и 10.32 мс
   при L512 в профилированном прогоне. Веса остаются FP32, а каждый вызов
   открывает/закрывает autocast. Это согласуется с повторным приведением весов;
   точную долю weight casts против activation casts отдельно не измеряли.
2. **Много мелких запусков:** 1259 kernels на коротком входе, 1139 при L512.
   В профилированном окне GPU занята ~29–61%; profiler сам увеличивает CPU overhead,
   поэтому это не доля простоя в обычном latency-прогоне.
3. **С ростом входа растут GEMM и attention.** Доли суммарного kernel time
   L16 → L512: matmul 28.0 → 33.1%, copy 58.3 → 33.2%,
   elementwise 8.7 → 17.6%, attention 1.9 → 13.8%.
   Attention уже использует efficient SDPA FP16, не math fallback.
4. SDK postprocessing/tokenization добавляют несколько миллисекунд,
   но разность независимых p50 не является точным разложением e2e.
   В forward-трейсах `aten::_local_scalar_dense` не обнаружен.

## Идеи ускорения — пока не измерены

| Идея | Основание | Что проверить перед применением |
|---|---|---|
| Дольше удерживать autocast scope и его weight cache в inference worker | сотни conversions на запрос | поведение cache, конкурентность, память и равенство SDK probabilities |
| Постоянно хранить часть/все веса в FP16 | убрать повторное приведение и сократить весовую память | LayerNorm/decision head, finite outputs и оценки на редких/обфусцированных запросах; не гарантировано эквивалентно |
| `torch.compile` / CUDA Graphs с фиксированными shapes | >1100 kernels на forward | graph breaks, padding/marker shapes, разогрев и сравнение вердиктов |
| Batching через SDK | batch=1 оставляет маленькие операции | throughput отдельно от per-request p95, одинаковое число вопросов |
| Слияние elementwise/RoPE/norm | доля мелких операций растёт | numerics и реальное уменьшение числа запусков |

Decode отсутствует: генерировать токены эта модель не должна.

## Проверки и ограничения

Smoke прошёл без NaN/Inf на реальных marker logits и action logits.
Для вопроса `danger`: безопасный запрос → B (P(A)=0.0829),
запрос про бомбу → A (P(A)=0.8459). Русский и base64-примеры не имеют
размеченного ожидаемого ответа; они не доказывают поддержку языков/обфускации.
SDK предупредил о clamp температуры bucket `choice:11+`; в эксперименте
используется другой bucket, `choice:2`.

Одна сессия, один синтетический текст, 50 замеров на длину; частоты/температура
GPU не контролировались. Kernel categories — эвристика. После прогона
исправлено ошибочное совпадение `cast` с `nocast` в классификаторе: доли и
top kernels пересчитаны **из полных исходных traces**, сами задержки не изменены.
Источник и дополнительные счётчики — `analysis.json`.

`latency_samples.csv` содержит 700 raw samples; все семь forward traces и
три phase traces проверены на наличие CUDA kernels. Phase traces размечают
модули и перенос результата, но не полный SDK postprocessing.
