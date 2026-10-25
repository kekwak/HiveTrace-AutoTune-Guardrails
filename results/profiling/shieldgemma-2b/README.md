# google/shieldgemma-2b — профилирование

**Выполнено в Colab 25.10.2026, одна Tesla T4.** PyTorch 2.11.0+cu130,
Transformers 4.57.1, batch=1, 2 CPU-потока. Фактические веса и вычисления —
**FP32**, что подтверждено `env.json`: `native_bf16=false`, `weight_dtype=torch.float32`.

Ноутбук: [`notebooks/shieldgemma_profile.ipynb`](../../../notebooks/shieldgemma_profile.ipynb).

- Revision `d1dffc9c8c9237a90aab09c61383791e718ef9e8`.
- Официальный chat template, prompt-only policy Dangerous Content.
- Официальный scoring: softmax последнего logits по vocab IDs `Yes`/`No`.
  Нет генерации объяснения; это штатный режим модели, не оптимизация decode.
- Batch=1, eager attention, `use_cache=False`, `logits_to_keep=1`.
  SDPA в Transformers 4.57.1 игнорирует attention softcapping Gemma2;
  eager сохраняет штатное вычисление (`attn_logit_softcapping=50`).
- BF16 на поддерживающей GPU; **FP32 на T4** как консервативный baseline
  вместо неподтверждённого FP16. Сравнение с FP16 соседних работ не равнозначно.
  Проверяется именно `torch.cuda.is_bf16_supported(including_emulation=False)`:
  в PyTorch 2.11 обычный вызов возвращает True на T4 из-за эмуляции.
- 5 прогревов, 50 замеров forward и e2e; текст 16–1024 токена плюс политика.
- Полный контекст 8192; превышение вызывает ошибку, а не скрытое усечение.
- Проверка finite logits обязательна. Smoke не является benchmark качества.

## Доступ и запуск
Доступ к `config.json` и индексу весов через локальный HF_TOKEN подтверждён.
Для Colab добавить `HF_TOKEN` в Secrets и разрешить notebook access:
локальное окружение не передаётся автоматически. При отказе проверить,
что аккаунт токена принял лицензию Gemma на Hugging Face.

Загрузить notebook в Colab, выбрать GPU, выполнить сверху вниз в свежей session.
Артефакты в `/content/results/profiling/shieldgemma-2b/<timestamp>/`:
среда, smoke, latency и raw samples, GPU-профиль, доли и топ CUDA-ядер,
Perfetto traces, итоговый report и ZIP. `manifest.json` со статусом completed
создаётся только в конце; большие traces не добавлять в git.

## Ключевые цифры

| Текст, токенов | Полный вход | forward p50 / p95, мс | scoring e2e p50 / p95, мс |
|---|---|---|---|
| 16 | 141 | 273.37 / 278.92 | 282.24 / 287.36 |
| 128 | 253 | 372.13 / 376.65 | 375.67 / 380.69 |
| 256 | 381 | 615.37 / 621.18 | 631.93 / 639.14 |
| 512 | 637 | 881.65 / 888.35 | 885.29 / 888.98 |
| 1024 | 1149 | 1618.87 / 1622.19 | 1621.40 / 1627.57 |

У benchmark-текстов шаблон добавляет 125 токенов. Пик allocated GPU памяти —
**9.85–10.03 GiB**, веса — 9.74 GiB / 2.614B параметров. Это не полный расход
`nvidia-smi`: CUDA context и зарезервированная allocator память не учитываются.
Пять warmup и 50 measured iterations на длину и режим, 700 raw samples.

## Узкие места

1. **FP32 matrix multiplication.** Matmul занимает 95.6% kernel time при
   L16 и 90.7% при L1024. Основные kernels — `volta_sgemm_*`; имя Volta —
   название ядра cuBLAS, а не другая GPU. В FP32 на T4 не используются
   быстрые FP16 Tensor Core GEMM. Категория matmul включает и линейные слои,
   и QK/AV attention; их отдельные доли этим классификатором не определяются.
2. **GPU почти непрерывно занята:** 96.8–99.5% внутри профилированного окна.
   1612 kernels на forward, но уменьшение CPU launch overhead само по себе
   здесь не выглядит главным источником ускорения. Profiler меняет абсолютные
   времена; проценты не вычислялись делением kernel time на обычный p50.
3. **Постоянная политика занимает большую часть короткого входа:**
   125 из 141 токена. При всех четырёх политиках отдельно стоимость будет
   иной; здесь измерялась только Dangerous Content.
4. **Eager attention обязателен для этого baseline.** В Transformers 4.57.1
   SDPA игнорирует аргумент softcap, а checkpoint требует
   `attn_logit_softcapping=50`. Переключить backend без проверки эквивалентности
   — значит изменить модель. `sdpa_backend=not detected` здесь ожидаем.
5. Copy — всего 0.6–0.7% kernel time, `aten::_to_copy` — 5 вызовов на forward,
   `aten::_local_scalar_dense` не обнаружен. Проблема повторного приведения
   весов, наблюдаемая у Laya, здесь отсутствует.

## Идеи ускорения — не измерены

| Идея | Какое ограничение снимает | Ограничения/проверки |
|---|---|---|
| L4/A100 с нативным BF16 | FP32 matmul без Tensor Cores на T4 | отдельный аппаратный baseline, не прямое сравнение на одной T4 |
| FP16 или смешанная точность на T4 | ускорить GEMM и уменьшить память весов | overflow/underflow, P(Yes), вердикты и редкие атаки; finite smoke недостаточен |
| Кэш постоянного префикса | не пересчитывать начало инструкции для каждого текста | новый KV-cache путь (`use_cache=False` в baseline), граница токенизации, masks, sliding window, softcap и численная эквивалентность |
| Fused attention с поддержкой Gemma2 softcap | eager materialization и отдельные elementwise | backend должен поддерживать softcap и архитектуру GPU; штатный SDPA не эквивалентен |
| Batching | увеличить throughput | не обещает меньший per-request p95, особенно на длинном входе |

Decode и генерация объяснения не измеряются: штатный scoring получает P(Yes)
из первого forward. Кэш префикса и дополнительные оптимизации не применялись.

## Smoke и ограничения

Проверки finite logits пройдены. P(Yes): безопасный запрос **0.000105**,
запрос про бомбу **0.989151**, пустой текст **0.005624**.
Русский и base64-примеры — лишь проверка выполнения без размеченного эталона.
Это не оценка качества или калибровки на датасете.

Одна сессия и один синтетический текст; p95 по 50 точкам приблизителен,
частоты/температура GPU не контролировались. Эвристика по именам ядер не
разделяет attention GEMM и linear GEMM. Низкая точность могла бы ускорить
вычисления, но из этого прогона нельзя утверждать отсутствие потери качества.

Сохранены CSV, среда, smoke и `analysis.json`. Все семь forward traces и
три phase traces проверены на наличие CUDA kernels; они лежат в локальном
игнорируемом архиве `docs/research/shieldgemma-fp32-20261025-073514.zip`.
Ноутбук генерирует traces заново. Разметка phase traces включает модули,
template/tokenize/H2D, forward и D2H; она не включает CPU softmax P(Yes).

Первый прогон 25.10.2026 завершился в BF16 через эмуляцию, а не в планируемом
FP32. Это установлено по фактическому dtype в `env.json`; его цифры не
выдаются за FP32 baseline. Архив сохранён локально в игнорируемом
`docs/research/shieldgemma-20261025-071902.zip`.
