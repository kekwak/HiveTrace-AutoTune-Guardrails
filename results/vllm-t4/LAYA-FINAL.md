# Laya на Colab T4: итог исследовательской проверки

## Решение

**Запускать decision-only Laya через vLLM технически возможно, но адаптер не делает её автоматически оптимальной.** Собственный encoder wrapper и decision pooler работают внутри engine, но по пропускной способности не показали устойчивого преимущества перед length-bucketed PyTorch reference. Для CLI разумно оставить независимый контракт модели и выбор backend, а не объявлять vLLM обязательным или оптимальным по умолчанию для Laya.

Это завершённая исследовательская проверка **FP32 decision-only пути**, не production release и не полная совместимость SDK. Action-head, confidence API, зарегистрированная публичная ReadStrategy, FP16/AMP, compile и fused/padded-batched реализация головы не входят в результат. Штатная mixed-precision SDK Laya не сравнивалась по скорости с этими FP32 путями. Вывод не распространяется на остальные модели.

## Что проверено

- Checkpoint `convaiinnovations/laya` revision `7b928d828b7b0e022f929d9bd2e44165aa270148`.
- vLLM 0.32.0, Transformers 5.19.0, PyTorch 2.13.0; CUDA 13.0 в первоначальных manifests.
- Custom model и pooler выполняют ModernBERT, type embedding, два TransformerEncoderLayer и marker scorer внутри vLLM.
- FP32 reference из pinned HF source, а не отдельный SDK package. Decision-only forward проверен на GPU против полного reference: `torch.testing.assert_close(atol=1e-6,rtol=1e-6)` на синтетическом choice:2. Batch-index grouping и восстановление порядка имеют unit tests.
- Исходный полный quality pass повторён на новом T4-выделении: оба раза 9711 outputs от каждого backend, ни одного failed batch и ни одной смены бинарного решения между backend.
- Performance sweep повторён на двух **разных физических GPU**: `GPU-56330e80-dae1-ab2e-6d9d-a8acef36af35` и `GPU-6364abcd-10da-33b4-44c4-09ea0e05cb79`. Один более ранний потерянный runtime не учитывается.

## Качество и усечение

Полные метрики по каждому dataset/split и объединению: `laya-dataset-report.json`, независимый повтор `laya-dataset-repeat-report.json`; краткие таблицы — [laya-dataset-summary.md](laya-dataset-summary.md). В обоих passes confusion counts совпали.

| Все 9711 записей | Значение, оба backend |
| --- | ---: |
| Coverage outputs | 100% |
| Accuracy | 59.16% |
| Precision harm | 39.87% |
| Recall harm | 50.06% |
| F1 harm | 44.39% |
| Усечённые inputs | 547 |
| Verdict flips reference → vLLM | 0 / 9711, в каждом из двух passes |

Качество невысоко для выбранного broad English safety-вопроса и root English checkpoint. Это **не универсальная оценка Laya**, не проверка multilingual checkpoint и не настройка prompt/threshold на тестовом наборе. Output оценивается без оригинального request context. 100% coverage не означает, что сохранён полный текст: опасный suffix может исчезнуть при штатном truncation. Сам перенос в vLLM проблему политики, языка или контекста не исправляет.

### Численная оговорка

Max raw-logit delta полного прохода — 0.024821; max probability delta — около 0.00304022. Выброс воспроизводится на втором GPU, на коротком input-robust/safe (94 токена, без PAD и truncation). В дополнительных опытах с тем же входом:

- cross-backend singleton probability delta ≈0.00261737;
- cross-backend duplicate-group delta ≈0.00533458;
- у самого reference singleton → исходная группа delta ≈0.00283390;
- у самого vLLM singleton → исходная группа delta ≈0.00241106;
- повтор singleton внутри каждого backend совпал; verdict flips в этих вариантах не было.

Таким образом, выявлена чувствительность к batching **в обоих** backend, но точная причина на уровне kernels/backbone/head не локализована. Нельзя утверждать строгую численную эквивалентность или сохранение калибровки при любом batching. Указанные цифры относятся к вероятностям вариантов, не confidence/action API. Агрегаты: `laya-outlier-diagnosis.json`, `laya-outlier-shapes.json`. Эти диагностические варианты не являются новым full-dataset quality pass.

## Равное decision-only сравнение скорости

Идентичные 288 заранее токенизированных inputs: по 48 равномерно выбранных по индексам из шести сплитов, без изменения selection после измерений. Hash input_ids и hashes всех четырёх исполняемых benchmark dependencies совпадают во всех сохранённых runs. Benchmark включает CPU collation, grouping, H2D, forward и CPU outputs/restoration; не включает сеть, tokenization или загрузку модели. Оба пути вычисляют только decision logits; reference всё ещё хранит небольшие action-head weights, но не выполняет action-head.

В каждом allocation/backend: batch 1/4/16/32 × mixed/bucketed order × три полных прохода. 15 секунд общего sustained warmup и два warmup batches на конфигурацию. Порядок конфигураций перемешан детерминированно по allocation; порядок backend обращён во втором allocation. CPU torch threads=2. Для vLLM client chunk не равен фактическому GPU batch: max_num_seqs=16, max_num_batched_tokens=2048. Голова в pooler исполняется по одной последовательности. Prefix caching и hybrid cache manager выключены; eager=True. Всего 96 successful timing trials, OOM/error нет.

Медианы трёх samples, records/s:

| Клиентский пакет | Reference bucketed, GPU 0 | vLLM mixed, GPU 0 | vLLM bucketed, GPU 0 | Reference bucketed, GPU 1 | vLLM mixed, GPU 1 | vLLM bucketed, GPU 1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 14.66 | 12.32 | 11.94 | 13.31 | 13.39 | 13.42 |
| 4 | 13.58 | 13.05 | 13.19 | 15.62 | 16.64 | 15.08 |
| 16 | 14.09 | 14.03 | 14.06 | 15.98 | 15.73 | 16.19 |
| 32 | 13.52 | 14.02 | 14.23 | 15.26 | 16.03 | 16.21 |

Лучшие наблюдённые конфигурации: reference 14.66 → 15.98, vLLM 14.23 → 16.64. Отношение best vLLM / best reference: **0.970 на GPU 0, 1.041 на GPU 1**. Это подбор лучшей точки того же небольшого sweep, не independent validation оптимума. При фиксированном bucketed batch=16 отношение около 0.998 и 1.013. Устойчивое преимущество vLLM не обнаружено; прежние «+17%» до выравнивания и bucketing не служат финальным выводом.

Raw trials и telemetry: `laya-fair-reference-{0,1}.json`, `laya-fair-vllm-{0,1}.json`. После последнего repeat каждой конфигурации проверены сохранённые outputs против reference mixed batch=1: 0 verdict flips на всех 16 snapshots по 288 inputs в каждом allocation, raw-logit delta до 0.000252843. Это не проверка каждой timing iteration и не full-dataset batch invariance.

### Память и аппаратный шум

Sampled device-wide memory.used максимум по полному sweep:

| Backend | GPU 0 | GPU 1 |
| --- | ---: | ---: |
| Reference | 5785 MiB | 6281 MiB |
| vLLM | 2185 MiB | 2213 MiB |

Это **дискретно измеренный максимум total device usage**, не точный tensor peak, не incremental memory и не сравнение памяти fastest конфигураций. Включает CUDA context и allocator reserve; после больших batches reserve сохраняется, поэтому сравнивать per-trial memory как независимую стоимость batch нельзя. Частота опроса — 0.25 с плюс время nvidia-smi; краткие пики могли быть пропущены. Вывод ограничен меньшим sampled maximum vLLM в данном sweep с различным scheduling/padding.

Температуры по spans: reference 50–80/70–82°C, vLLM 69–81/50–81°C; частоты заметно менялись (см. raw telemetry). 15 секунд warmup не доказывают steady state. Следовательно, точность малых процентов ускорения ограничена thermal/order effects и Colab variability; confidence interval устойчивого speedup не вычислялся. Два разных GPU UUID не делают малую выборку универсальным benchmark T4.

## Как использовать результат

1. Для CLI Laya — отдельный model adapter и backend interface. vLLM допустим как экспериментальная реализация, но не обязателен ради скорости.
2. Для пакетного production baseline сначала рассмотреть length bucketing, token budget и explicit truncation policy. Не превращать measured throughput в SLA.
3. Перед production: validation metadata до engine, полный weight/config contract, нужные API поля, загрузка без ручного checkpoint conversion, dependency pinning, error isolation, проверка на production distribution. Эти пункты здесь **не выполнены** и не являются скрытым обещанием готового CLI.

## Воспроизведение

Эксперименты выполнены в отдельном worktree без изменения исходной рабочей ветки; код и агрегаты оформляются отдельным PR после проверки. Секретов в report нет. Private parquet, tokenized inputs и per-record logits не скачивались из disposable runtime.

В свежей Colab T4 установить `vllm==0.32.0`; передавать read token datasets в `DATASET_TOKEN` только через секретное окружение, не в notebook source или CLI arguments. Веса Laya публичны. Загрузить в `/content/` модули `laya_dataset_probe.py`, `laya_vllm_model.py`, `laya_fair_benchmark.py`, `batching.py`, `metrics.py`, `reporting.py`. Исполнять отдельными Python processes:

```text
python /content/laya_dataset_probe.py prepare
python /content/laya_fair_benchmark.py reference 0
python /content/laya_fair_benchmark.py vllm 0
```

Скачать `fair-reference-0.json` и `fair-vllm-0.json` из `/content/laya-dataset/` сразу после каждого backend. Остановить runtime. Во втором новом T4 allocation поменять порядок backend и label на 1. Quality repeat: отдельные processes `laya_dataset_probe.py reference`, затем `vllm`, затем `report`; скачивать только `aggregate-report.json`. Не использовать голый `runpy.run_path()` как последнее выражение интерактивной ячейки — словарь locals может раскрыть private inputs.

Hashes точных fair-benchmark sources, GPU UUID, stack versions и samples включены в каждый fair report. Первые исторические quality runs имели более слабую source provenance; они не объявляются эквивалентными новому runner побайтно. Полный deployment lockfile и wheel hashes не сохранены.
