# HaloGuard1-Gen-0.8B: поиск узких мест и проверка гипотез

Подготовлено 1 ноября 2026 по историческому прогону
`20261025T222702Z_274343fd` и опубликованным материалам.
Это дополнение к результатам ускорения: объясняет, почему исследовали linear
attention и FLA. Новых GPU-замеров и визуального просмотра Perfetto не выполнялось.

## Краткий вывод

Главный кандидат в исследованном FP16 reference-пути — реализация linear
attention с большим числом мелких GPU-запусков. Замена только этого пути на
FLA резко сокращает число kernels и в данной сессии снижает обычную end-to-end
latency. Это поддерживает гипотезу о неэффективном исполнении reference-пути,
но не доказывает, что всё время тратилось именно на CUDA launch или что модель
в целом launch-bound. Full attention не был первым кандидатом по этому профилю.

## Условия и доказательства

- Checkpoint: `astroware/HaloGuard1-Gen-0.8B`, revision
  `85ff68dd063a7e58487a4251824fd36a20977d5c`.
- Kaggle, одна T4, batch 1; Python 3.13.15, PyTorch 2.11.0+cu128.
  Точные зависимости и GPU-паспорт: [environment.json][environment].
- Контроль `P1`: FP16 eager, reference linear attention. Кандидат `H2_linear`:
  FP16, только готовый FLA linear-attention path; conv/norm и full attention
  не заменялись пакетом дополнительных оптимизаций.
- Авторский промпт и scoring safe/unsafe с порогом 0,65 сохранены. Полный
  контракт, входы и результаты доступны в [каталоге запуска][run].
- Legacy short/medium/long: 32/128/384 пользовательских и 429/525/781 полных
  токенов. Длины проверены по сырым latency-строкам; это не новые бакеты B1–B4.
- Профили short/long записаны отдельно от latency, по три итерации.
  Latency: пять парных блоков по 100 запросов на сценарий, семь сценариев,
  две границы тайминга, A/A-контроль отдельно. Это одна подтверждающая сессия,
  не завершённое независимое подтверждение на трёх VM.

## Что видно в профиле

Ниже программная сводка сохранённой связи CPU operator → GPU kernel и областей
модулей. Число kernels дано суммарно за три итерации; время компонента — средняя
сумма длительностей его kernels на итерацию. Это **не wall time, не SM utilization
и не декомпозиция обычной latency**. Категория `copies` не означает автоматически
PCIe/H2D: она включает операции копирования тензоров внутри GPU-вычислений.

| Профиль | Все kernels, за 3 итерации | Linear-attention kernels, за 3 итерации | Linear attention, мс/итерацию | Copies, мс/итерацию | Full attention, мс/итерацию |
| --- | ---: | ---: | ---: | ---: | ---: |
| P1 short | 43 917 | 27 648 | 60,21 | 23,70 | 4,97 |
| FLA short | 7 114 | 2 160 | 81,57 | 8,16 | 4,88 |
| P1 long | 49 638 | 33 047 | 92,99 | 38,03 | 10,84 |
| FLA long | 7 005 | 2 052 | 153,01 | 16,25 | 12,42 |

1. У reference-пути десятки тысяч запусков; linear attention составляет большую
   часть их числа. Это конкретное основание искать готовую реализацию этого блока,
   а не начинать с произвольной оптимизации всего Transformer.
2. После FLA количество запусков сильно уменьшается. В атрибуции появляются
   `ChunkGatedDeltaRuleFunction` / `chunk_fwd_kernel_o`; это свидетельство исполнения
   нужной реализации, а не только наличия установленной библиотеки.
3. Суммарное GPU-время linear attention под профайлером после FLA **увеличивается**,
   хотя обычная latency уменьшается. Поэтому объяснение «все kernels стали быстрее»
   неверно. Наблюдения совместимы с меньшими накладными расходами исполнения;
   точную долю CPU dispatch/launch без отдельного анализа приписывать нельзя.
4. Full attention занимает меньшую сумму kernel-времени, чем reference linear
   attention и связанные операции. Это основание для порядка экспериментов,
   но не формальная верхняя граница speedup по закону Амдала.

Сводные значения, SHA256 исходной атрибуции и опубликованных traces:
[evidence.json](evidence.json). Неатрибутированные события остаются `unknown`.
Составные fused operators не разлагаются на copy/norm по подстроке имени.

## От наблюдения к эксперименту

| Наблюдение | Гипотеза | Проверка | Что установлено |
| --- | --- | --- | --- |
| Много мелких kernels в reference linear attention | Готовый специализированный путь сократит накладные расходы | P1 против H2_linear, одна точность и одинаковые входы | В исторической сессии число запусков и latency снизились |
| Full attention не доминирует в kernel-сводке | SDPA-only даст меньший эффект | Отдельный SDPA screening | По опубликованному итогу полезное ускорение полного запроса не установлено; это не универсальная бесполезность SDPA |
| Даже после FLA linear attention остаётся заметным | Возможны другие реализации/сочетания | Отдельные сравнения backend и сочетаний | Здесь не проверены; не объявляем FLA глобально лучшим вариантом |
| В reference есть копирования и операции norm/conv | Возможна дополнительная fusion | Изолированные изменения с новой регрессией | По этому прогону самостоятельный эффект не установлен |

FLA — замена реализации вычислительного блока, не vLLM-сервинг, не квантование
и не prefix caching. Сравнение с готовыми serving-средами в этом отчёте отсутствует.

## Как проверили гипотезу без профайлера

Медианы пяти блочных p50/p95 полного пути «текст → вердикт», миллисекунды.
Это не pooled-квантили и не отношение парных квантилей.

| Сценарий | p50 P1 → FLA | p95 P1 → FLA |
| --- | ---: | ---: |
| legacy_short | 349,28 → 126,42 | 364,05 → 127,69 |
| legacy_medium | 373,39 → 159,58 | 386,69 → 160,73 |
| legacy_long | 428,35 → 232,00 | 449,36 → 233,90 |

Все сценарии, парные отношения и шум: [end_to_end.csv][latency]. На 300 отложенных
синтетических примерах опубликовано отсутствие смен вердиктов и max Δp_unsafe
0,002340 для FLA относительно FP32. Примеров около порога не было.
Это **численная регрессия, не оценка качества или безопасности**.
Подготовка и прогрев вынесены из latency; результат не доказывает окупаемость
холодного старта. FP16-контроль не сравнивается с историческим FP32 из другой сессии.

## Как проверить самостоятельно

- [P1 short][ps] и [FLA short][fs]; [P1 long][pl] и [FLA long][fl]. Скачать gzip,
  распаковать при необходимости и открыть в Perfetto. Сравнивать одинаковые длины,
  выбрать одну из трёх итераций, смотреть module scopes и CPU–GPU correlation.
- [Исходники атрибуции][code] и [проверки backend/численных ответов][checks].
- Все восемь traces двух моделей в этом пакете сверены по gzip/raw SHA256;
  файлы не переписывались. Подробности и сводки рядом в evidence.json.

## Ограничения и правило-кандидат

Правило-кандидат: для **данной версии HaloGuard, T4 и измеренного FP16 reference-пути**
проверять FLA linear attention как отдельную замену с фиксированными prompt/scoring,
контролем backend и численной регрессией. Эффект зависит от длины и среды.
Не переносить старые проценты на новые зависимости, корпус, RTX 4090/5090 или другой
контроль. Отдельные последующие измерения не заменяются этим историческим отчётом.
Нет аппаратных счётчиков для строгого вывода compute/memory/launch-bound;
нет визуальных иллюстраций Perfetto, quality-набора или доказанного safety-сохранения.

[run]: https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/tree/7f07499df15f28b2452406995630a97cda211d4b/results/optimization/haloguard-fla
[environment]: https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/blob/7f07499df15f28b2452406995630a97cda211d4b/results/optimization/haloguard-fla/environment.json
[latency]: https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/blob/7f07499df15f28b2452406995630a97cda211d4b/results/optimization/end_to_end.csv
[checks]: https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/blob/7f07499df15f28b2452406995630a97cda211d4b/results/optimization/haloguard-fla/numerical_and_backend_checks.json
[code]: https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/blob/7f07499df15f28b2452406995630a97cda211d4b/u1/experiments/evidence.py
[ps]: https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/blob/7f07499df15f28b2452406995630a97cda211d4b/results/optimization/haloguard-fla/traces/P1_legacy_short.json.gz
[pl]: https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/blob/7f07499df15f28b2452406995630a97cda211d4b/results/optimization/haloguard-fla/traces/P1_legacy_long.json.gz
[fs]: https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/blob/7f07499df15f28b2452406995630a97cda211d4b/results/optimization/haloguard-fla/traces/H2_linear_legacy_short.json.gz
[fl]: https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/blob/7f07499df15f28b2452406995630a97cda211d4b/results/optimization/haloguard-fla/traces/H2_linear_legacy_long.json.gz
