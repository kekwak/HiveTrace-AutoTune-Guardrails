# Laya: reference против custom vLLM adapter на T4

**Исторический первый прогон.** Итог после decision-only выравнивания, bucketing и двух физических T4: [LAYA-FINAL.md](LAYA-FINAL.md). Прежние +17% не являются окончательным выводом.

Статус: исследовательский FP32-прогон, без квантования. Encoder и decision head находятся внутри engine. Проверяются raw decision logits и calibrated binary probabilities; action-head, полный SDK contract и `/v1/systemone` не реализованы.

## Протокол

Pinned checkpoint и datasets, вопрос и варианты, выбранная temperature и ограничения сохранены в `laya-dataset-report.json`. Используется один broad safety choice-вопрос на английском, root English Laya, а не multilingual checkpoint. Для output оценивается отдельно текст ответа без исходного запроса. Соответствие общей safety политики всем категориям датасетов не установлено. Это baseline данного prompt, не универсальная оценка качества Laya.

Оба backend получили идентичные заранее токенизированные последовательности. Reference: FP32 weights/compute, batch=4, включая act_head. vLLM 0.32.0: FP32, eager, без prefix caching и hybrid cache manager, client chunks=32, max_num_seqs=16 и max_num_batched_tokens=2048, голова обрабатывает последовательности по одной. Входы усечены штатным build_sequence до 512; это отличается от пропуска длинных строк в GLiNER experiment. Tokenizer/reference взяты из pinned HF source, не из установленного SDK 0.3.20.

## Качество

Все 9711 записей обработаны обоими backend, failed batches=0, метрики идентичны, verdict flips=0/9711. Coverage=100% означает полученный output, **не отсутствие усечения**. Усечены 547 записей. Max calibrated probability delta=0.003040254 на input-robust/safe; отсутствие flips не означает побитовую идентичность или одинаковую калибровку.

После независимого ревью отчёт пересчитан с проверкой точного количества outputs, единым валидатором двух конечных logits и сравнением raw logits. Max raw-logit delta=0.024821043 на input-robust/safe; на остальных сплитах maxima от 4.09e-5 до 2.06e-4. Причина выброса отдельно не установлена, численная эквивалентность по строгому допуску не заявляется. Пересчёт не изменил quality metrics и нулевое число flips. Значение probability delta в новой версии JSON вычислено стабильным softmax на Python float и может незначительно отличаться от первоначального FP32 расчёта.

| Датасет | Сплит | N | Accuracy | Усечены |
| --- | --- | ---: | ---: | ---: |
| input | safe | 2550 | 71.57% | 161 |
| input | harm | 345 | 35.36% | 8 |
| input-robust | safe | 3599 | 58.68% | 1 |
| input-robust | harm | 2756 | 51.12% | 78 |
| output | safe | 400 | 56.25% | 269 |
| output | harm | 61 | 85.25% | 30 |

| Набор | N | Accuracy | Precision harm | Recall harm | F1 harm |
| --- | ---: | ---: | ---: | ---: | ---: |
| input | 2895 | 67.25% | 14.40% | 35.36% | 20.47% |
| input-robust | 6355 | 55.41% | 48.65% | 51.12% | 49.86% |
| output | 461 | 60.09% | 22.91% | 85.25% | 36.11% |
| Все записи | 9711 | 59.16% | 39.87% | 50.06% | 44.39% |

Общий набор — объединение примеров, не среднее метрик сплитов. Полные confusion counts, specificity и одноклассовые метрики находятся в JSON. Невысокое абсолютное качество одинаково у обоих backend; перенос не улучшает suitability выбранного prompt/checkpoint. Сравнение с GLiNER неравнозначно из-за разных политик, ограничений и покрытия.

## Пакетный throughput

Фиксированная сбалансированная выборка: 48 равномерно по индексам из каждого из шести сплитов, всего 288. Исходные lengths неодинаковы, штатное усечение сохраняется. Обоим backend передаются те же pretokenized inputs, измерение включает CPU batching, H2D, forward и CPU outputs, но не загрузку модели, скачивание и токенизацию. Reference вычисляет action-head, vLLM его не вычисляет. По одному warmup batch перед каждой конфигурацией, затем три полных прохода; без рандомизации порядка backend/batch, без telemetry и независимой GPU allocation. Steady state не доказан.

| Client batch size | Reference, median records/s | vLLM, median records/s |
| ---: | ---: | ---: |
| 1 | 15.07 | 14.47 |
| 4 | 13.67 | 16.16 |
| 16 | 11.51 | 17.32 |
| 32 | 9.63 | 17.65 |

По три успешных измерения на точку, OOM не было. В vLLM client chunk не равен фактическому GPU batch. Reference использует padding смешанных длин без length bucketing; это не оптимизированный baseline. Лучший vLLM примерно на 17% выше лучшего reference в этом sweep. Это **не изолированная стоимость engine, не утверждение об устойчивом ускорении и не доказательство оптимальности**. Применение length bucketing и одинакового decision-only head к reference может изменить вывод. Полные samples: `laya-reference-benchmark.json`, `laya-vllm-benchmark.json`. Peak memory для этого сравнения не измерена.

Однократный полный quality pass занял 430.92 с у reference и 345.40 с у vLLM; разные batches, без повторов. Эти timings включают сериализацию и запись private logits, не являются inference-only и не подменяют таблицу выше.

## Воспроизводимость и безопасность

Код: `src/vllm_probe/laya_dataset_probe.py`, `laya_vllm_model.py`, `metrics.py`. Environment и hashes финальных скриптов: `laya-dataset-environment.json`; benchmark branch добавлен после quality pass, поэтому hash финального dataset runner не заявляется точным hash первого исполнявшегося источника. Private texts и tokenized inputs не скачивались, в отчётах только агрегаты. Не публиковать промежуточные private inputs/logits. Disposable runtime необходимо остановить после выгрузки.

Независимое ревью выявило и исправило silent zip truncation и отсутствие dataset-wide raw-logit comparison. Regression tests проверяют count mismatch, malformed outputs и общий offset logits при одинаковых probabilities. Локальный suite — 18 passed; GPU reports пересчитаны из существующих private outputs без повторного inference. Тесты не покрывают production readiness адаптера и все возможные условия vLLM scheduling.
