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

Исходные traces доступны после clone: [`traces/baseline.zip`](traces/baseline.zip),
контрольная сумма — [`traces/SHA256SUMS`](traces/SHA256SUMS).
Воспроизводимый пересчёт corrected kernel shares, операторной атрибуции и
визуализаций: [`src/profiling/README.md`](../../../src/profiling/README.md).

Аннотированные timelines из исходных events (не screenshots интерфейса Perfetto):
[L16](figures/timeline_L16.svg), [L512](figures/timeline_L512.svg).

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

## Гипотезы ускорения и статус проверки

| Идея | Основание | Что проверить перед применением |
|---|---|---|
| `no_grad` + persistent autocast, отдельно от `inference_mode` | сотни conversions на запрос | Проверено: conversions 319→183, но стабильное wall-time ускорение не доказано; цена — дополнительный weight cache |
| Постоянно хранить часть/все веса в FP16 | убрать повторное приведение и сократить весовую память | Проверено: `.half()` без AMP падает; FP16+AMP сокращает память, но не показал устойчивого ускорения и меняет probabilities |
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

Исходный baseline — одна сессия, один синтетический текст, 50 замеров на длину;
частоты/температура GPU в нём не контролировались. Дополнительные независимые
повторы и A/B сохраняются отдельно в `validation/`, не заменяют исходные samples.
Kernel categories — эвристика. После прогона
исправлено ошибочное совпадение `cast` с `nocast` в классификаторе: доли и
top kernels пересчитаны **из полных исходных traces**, сами задержки не изменены.
Источник и дополнительные счётчики — `analysis.json`; воспроизводимый пересчёт
по архиву — `src/profiling/analyze.py`, SHA256 всех файлов и аудит — `trace_audit.json`.

`latency_samples.csv` содержит 700 raw samples; все семь forward traces и
три phase traces проверены на наличие CUDA kernels. Phase traces размечают
модули и перенос результата, но не полный SDK postprocessing. CPU ranges
асинхронны: D2H span может ждать предшествующий forward, а не только копировать.
В follow-up снимается `actual_e2e.json` настоящего SDK-пути и CUDA Events
forward interval; CPU/D2H range не трактуется как самостоятельная стоимость.

Исправление рекомендации про cache основано на
[PyTorch v2.11.0 `cached_cast`](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/autocast_mode.cpp):
условие кэширования включает `!c10::InferenceMode::is_enabled()`.

## Контроль после независимого ревью

### Три независимых kernel runs

Новая T4 VM, три свежих ядра; по 400 raw samples на run, **1200 всего**.
Общий прогрев 40 запросов, затем 20 warmup + 50 samples на case/mode.
Порядок восьми case/mode блоков перемешан независимо в каждом run.
Проверены два разных текста — benign и danger_request — при одинаковых
фактических длинах 16/512 текстовых токенов. Длинные Laya-входы по-прежнему усечены.

| Case | SDK e2e p50: диапазон между runs, мс | p95: диапазон между runs, мс |
|---|---|---|
| benign L16 | 33.56–35.97 | 41.51–48.72 |
| benign L512 | 41.73–42.30 | 45.95–47.05 |
| danger_request L16 | 33.74–34.83 | 42.49–48.66 |
| danger_request L512 | 41.04–42.29 | 49.75–55.21 |

Полные forward/e2e агрегаты — [`validation/repeat_aggregate.csv`](validation/repeat_aggregate.csv).
Секундная telemetry, idle/peak/incremental allocated и reserved memory сохранены.
Диапазоны — **не confidence interval production p95**; три ядра одной VM не равны
трём независимым GPU. Telemetry включает загрузку, reference, idle и A/B, поэтому
минимальные clocks всего файла не описывают clocks конкретного latency-блока.

### Precision/cache A/B — измерен, не обещан

Ниже отдельный run-0, direct forward, **не e2e** и не те же samples, что baseline.
Режимы измерялись блоками в случайном порядке, а не попарно чередовались на каждом
запросе; маленькие различия и причинность wall-time эффекта остаются неопределёнными.

| Режим | `_to_copy`/forward | p50 L16 / L512, мс | peak allocated L16, GiB |
|---|---|---|---|
| inference_mode, autocast на вызов | 319 | 32.09 / 44.96 | 1.59 |
| inference_mode, внешний autocast | 319 | 40.20 / 41.46 | 1.59 |
| no_grad, внешний autocast | 183 | 41.16 / 43.09 | 2.27 |
| FP16-веса + autocast | 251 | 43.26 / 44.52 | 0.82 |
| FP16-веса без AMP | — | ошибка Float/Half | — |

Чистый `.half()` не является готовым решением: `DecisionModel.forward` приводит
pooled/features к FP32, затем передаёт их в FP16 `act_head`; без autocast это
несовместимые dtype. Неудачный кандидат записан в `validation/run-0/ab_failures.json`.

Targeted traces с shapes подтвердили именно **weight casts**: `[5248,1024]`,
`[3072,1024]`, `[1024,2624]` — формы MLP/attention matrices. В no_grad/persistent
режиме эти большие copy kernels исчезают, остаются activation/mask copies.
L16 copy kernel time под profiler: **9.29→0.82 мс**, kernels/forward **1259→1123**.
Это подтверждает механизм cache, **но не гарантирует wall-time ускорение**:
в обычных samples уверенного выигрыша не получили. Разница profiler timings
не вычитается из обычного p50. Следовательно, casts — доказанная GPU-стоимость,
но главная причинная составляющая обычного wall time пока не установлена.

Результаты: `validation/run-0/ab_summary.csv`, `ab_samples.csv`,
`validation/ab_copy_shapes.csv`, `ab_profile_summary.csv`; полные targeted traces
в `validation/run-0/artifacts.zip`. Восстановление между кандидатами включает
nonpersistent RoPE buffers: одного state_dict после `.half()` недостаточно.

### Численные и контекстные проверки

- **11 sanity cases**, включая длинные benign/danger, обфускацию и русский:
  baseline mixed precision против FP32 reference, max |ΔP(A)| **0.001540**,
  смены argmax не обнаружены. Это не размеченный quality benchmark.
- Direct forward совпал с округлёнными SDK probabilities в пределах **0.0002**.
- FP32-weight cache/context candidates дали те же probabilities, что baseline;
  FP16+AMP: max |ΔP(A)| относительно baseline **0.001437**, относительно FP32
  **0.002059**, смены argmax на этих 11 cases не обнаружены.
- 477/478/479 текстовых токенов → 511/512/512 общего входа;
  только последний случай усечён. Опасный suffix после 478 токенов полностью
  исчезает: input_ids совпали с benign prefix. Это демонстрация границы,
  не доказательство способности модели обнаружить атаку.
- Новый `actual_e2e.json` вызывает настоящий `agent.predict`; отдельно сохранены
  CUDA Events forward intervals. CPU ranges остаются асинхронными.

Исходные и follow-up samples не смешиваются. Полные follow-up ZIP, SHA256,
численные проверки и telemetry доступны в [`validation/`](validation/).
Точный исходник validation runner в завершённых прогонах не сохранялся;
текущий скрипт не заявляется идентичным исполненному. Ограничение отмечено
в `validation/provenance.json`; будущий runpy-рецепт сохраняет source и SHA256.
