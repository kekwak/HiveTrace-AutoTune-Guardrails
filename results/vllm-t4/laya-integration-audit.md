# Laya: аудит custom integration vLLM

Источник: `convaiinnovations/laya` revision `7b928d828b7b0e022f929d9bd2e44165aa270148`, файлы `rl_common.py`, `rl_agent_api.py`, `rl_agent_config.json`; vLLM main `entrypoints/openai/decisions/strategies.py`, `model_executor/models/clef_flash.py`. Это аудит исходников, не выполненная интеграция и не подтверждение эквивалентности SDK 0.3.20.

## Что нужно сохранить

- Точная подготовка `[CLS] type question instructions [SEP] [MASK] option ... [SEP] state [SEP]`, порядок вариантов, очистка mask-токенов и budgets: max_len=512, head_max_len=192. Усечение state зависит от длины вопроса и вариантов.
- ModernBERT-large encoder, затем type embedding по каждому токену.
- **Два полных TransformerEncoderLayer над последовательностью** перед извлечением marker representations. Это не обычный linear classification head и не готовый Clef JointSchemaHead.
- Scorer на позициях MASK, mask несуществующих options, FP32 logits.
- Temperature по типу и количеству вариантов; для choice:2 в config это 1.9063563346862793, а не общая choice temperature.
- Если сохраняется весь API: act_head с CLS representation и четырьмя признаками распределения, confidence = 1 - normalized entropy. Это отличается от confidence текущего generative structured-decisions API vLLM (probability over vocabulary).
- Штатный root wrapper на T4 использует FP32 веса и FP16 autocast. Проверить отдельно актуальный пакет SDK, его дополнительные калибровки и clamp; HF wrapper не автоматически тождественен SDK.

## Архитектура адаптера — техническая гипотеза

1. Зарегистрировать config/model и загрузить checkpoint с точным mapping всех весов.
2. Подключить ModernBERT vLLM только после проверки encoder hidden states против reference.
3. Реализовать две дополнительные bidirectional attention layers с границами отдельных последовательностей. Нельзя объединять токены разных запросов во взаимно видимую последовательность. Вариант с padding обратно в batch и штатным PyTorch head проще для первой проверки, но теряет часть возможного выигрыша.
4. Передавать marker positions, qtype и option counts через поддержанный metadata/pooling contract; собственный pooler возвращает marker logits (и action outputs, если API это требует).
5. Собственный ReadStrategy выполняет точный рендер, temperature scaling и порядок outputs. Поддержка `/v1/systemone` сама по себе не загружает отсутствующую модель.

## Проверки перед выводом

Совпадение input IDs и marker positions; hidden states; logits до temperature; calibrated probabilities; verdict flips; act probability; confidence semantics; padding и batch invariance; truncation boundary 477/478/479 из существующего протокола; разные типы и количество опций. Затем throughput на одной T4 при одинаковой точности, прогреве и данных. До этих проверок утверждение «адаптер достаточен и автоматически оптимален» не подтверждено.

## Вывод

Structured decisions в vLLM дают подходящий **механизм расширения**, а не готовую поддержку Laya. Нужна полноценная model integration плюс стратегия чтения; возможность и скорость этого пути пока не измерены. Отказ стандартного загрузчика не является доказательством невозможности такой интеграции.

## Проверенный контракт передачи metadata

В исходниках vLLM main `JointSchemaPooler.forward` вызывает `self.pooling(hidden_states, pooling_metadata)`, затем перебирает отдельные последовательности вместе с `pooling_metadata.pooling_params`. Для каждой доступны `params.extra_kwargs` и собственная длина `hidden.shape[0]`. Значит, marker positions и qtype могут передаваться через request metadata без смешивания индексов разных запросов.

Экспериментальный контракт: `extra_kwargs={"laya": {"marker_positions": [3, 7], "qtype": 0}}`. Позиции абсолютные относительно подготовленного input_ids; порядок не сортируется, поскольку определяет порядок ответов. Дубликаты, отрицательные и выходящие за prompt_len позиции отклоняются. Отсутствующая metadata у profiling dummy возвращает None, но явно повреждённая metadata должна отклоняться ещё на входе, а не падать в engine core. Это локальный parser, **не зарегистрированный vLLM adapter**. Проверка фактических MASK token IDs, checkpoint mapping и выполнение head пока отсутствуют.

Тесты `tests/test_laya_metadata.py`: восемь случаев metadata, включая dummy; вместе с тестами бинарных метрик локальный suite: 10 passed. GPU equivalence и throughput не проверялись.

## Выполненный encoder smoke на T4

Из pinned Laya checkpoint выделены 170 `encoder.*` tensors из 206; prefix снят, config изменён с `ModernBertForMaskedLM` на `ModernBertModel` для отдельного encoder probe, tokenizer скопирован из checkpoint. Transformers AutoModel загрузил encoder strict=True без пропусков. Это не преобразование полной модели Laya: остальные 36 tensors головы остаются вне probe.

На vLLM 0.32.0/T4 выполнен `LLM(runner="pooling", pooler_config={"task":"token_embed"}, dtype="float32", max_model_len=512, enforce_eager=True)` и `encode(pooling_task="token_embed", pooling_params=PoolingParams(use_activation=False))`. Два синтетических текста, идентичные input_ids: outputs [8,1024] и [9,1024], finite=True. Triton backend выбран после исключения FA2 на sm75.

Сравнение против FP32-весов Transformers с FP16 autocast: max absolute delta 0.09921455 и 0.00647926, mean delta 0.00149725 и 0.00067642. **Точность вычислений различается**, поэтому эти значения не доказывают ни эквивалентность, ни дефект vLLM; сравнение на одинаковой точности и влияние на decision logits ещё не выполнены. Синтетические тексты не включают question markers и не оценивают качество модерации. Замеры throughput отсутствуют.

Технический вывод: препятствие standard loader обходится выделением encoder, и encoder inference на T4 работает. Это усиливает возможность custom integration, но полная Laya пока не работает внутри vLLM.

Повтор с одинаковой точностью FP32: Transformers reference вычислен отдельно для каждого текста без padding; vLLM получил оба текста одним encode. Max absolute delta: 0.0001773834 и 0.0000114441; mean delta: 0.00000241185 и 0.00000095479. Cosine в FP32 округлился чуть выше 1, поэтому не интерпретируется как точное математическое значение. Результат подтверждает близость encoder hidden states на этих двух коротких примерах, не на полном наборе и не на голове Laya. Все engine subprocess завершились с кодом 0.

## Выполненная гибридная проверка головы

`src/vllm_probe/laya_head_probe.py` — исследовательский GPU-скрипт (не production adapter). Pinned HF reference загружается strict=True; все веса приведены к FP32. Для восьми синтетических пар question/state сначала вычислен полный reference forward. Затем 170 encoder tensors выделены в ModernBERT checkpoint для vLLM; все восемь input_ids переданы одним encode, без output normalization. Hidden states возвращены на CPU и снова на GPU, после чего выполнены **штатные** type embedding, два TransformerEncoderLayer, marker scorer и act_head из reference. Таким образом, обмен данными и head остаются вне vLLM engine; этот путь не доказывает ускорение.

Успешный subprocess exit=0. Восемь случаев включают choice:2 и score:3, два английских текста, русский текст, повторяющийся state с усечением до 512. Все decision/action logits finite. Max decision-logit delta=9.7156e-6; max calibrated probability delta=1.3113e-6; verdict flips=0/8. Max action-logit delta=0.0021973 — action probabilities, confidence и threshold flips не проверены. Noul, иные cardinalities, adversarial input и полный датасет не проверены. Калибровка взята из HF config, не из отдельной реализации SDK. Report: `laya-head-report.json`.

Вывод ограничен технической осуществимостью гибридного пути: encoder vLLM совместим с reference head на этих случаях. Реализация head внутри pooler и зарегистрированная ReadStrategy отсутствуют. GPU performance sweep и полные quality metrics Laya по всем benchmarks по-прежнему отсутствуют.

## Следующий эксперимент: голова внутри engine

`src/vllm_probe/laya_vllm_model.py` регистрирует экспериментальную `LayaForDecisions`: ModernBERT encoder и собственный pooler с type embedding, двумя TransformerEncoderLayer и marker scorer. Pooler обрабатывает каждую последовательность отдельно, возвращает сырые logits вариантов через token_classify. **Action-head и temperature не загружаются намеренно**, калибровка и публичная ReadStrategy не реализованы. Это ограниченный decision-logit adapter, а не полная замена SDK. Нельзя использовать с недоверенной metadata: входная валидация parser пока не подключена к engine API. Поддержан только FP32.

В `laya_engine_probe.py` сравниваются восемь синтетических случаев с полным reference; question wording choice отличается от предыдущего гибридного probe, поэтому это не одни и те же численные cases. Engine возвращает восемь outputs с правильным числом вариантов, все finite; max raw-logit delta=9.9539757e-6, verdict flips=0/8. Проверка явно требует delta<0.001 и отсутствие flips; subprocess exit=0. Данные: `laya-engine-report.json`. Ни один output не вычисляется клиентской головой.

Промежуточные неудачи: positional vllm_config конфликтовал с compiler decorator (исправлено на keyword); default hybrid cache coordinator падал с отсутствующей cacheable group. Для bidirectional encoder probe явно выключены `enable_prefix_caching=False` и `disable_hybrid_kv_cache_manager=True`, после чего запуск прошёл. Это не проверка всех режимов scheduling/caching.

Теперь подтверждена техническая возможность выполнить encoder и decision scorer Laya внутри vLLM на T4. Автоматальная оптимальность по-прежнему не подтверждена: отсутствуют performance sweep, quality на benchmarks, noul, ошибки metadata, action API, ReadStrategy и независимое ревью адаптера.
