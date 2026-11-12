# У1: запуск измерителя v4

Актуальный CLI принимает только `schema_version=4`. Форматы результатов, данных и
протокола версионируются отдельно. Старые архивы и конфиги отклоняются явно;
для них остаётся историческая версия Git, миграция не входит в этот этап.
[Протокол](measurement_protocol_v1.md) · [пресеты](../configs/v4) ·
[история проверок](u1_handoff_readiness.md).

Baseline восьми пресетов соответствует [исполнению авторов HF/SDK](baseline_modes.md#schema-4-исполнение-авторов-hfsdk-12-ноября)
в закреплённой среде. Open/Hive больше не принуждаются к SDPA math;
Halo/Open/Hive/Shield используют BF16, Laya сохраняет native SDK AMP без
принудительного efficient kernel. Shield оставляет eager для сохранения soft-capping.
Фактические attention/precision и kernels записываются.
[Open BF16/native прошёл 4/4 верхних smoke-ячеек на RTX 5090](../results/u1_handoff/2026-11-12-v4-native-open/README.md).
Остальные новые режимы этим не проверены. OOM прежнего Open FP16+math не является
пределом нового baseline. Размер батча при ошибке не уменьшается автоматически.

## Быстрый старт: одна модель

Все команды ниже выполняются **из корня клона этой ветки**, на Linux x86_64
с Python 3.13. Начните со своей модели, не с пересборки всех восьми индексов.
Пример ниже использует Koala; [пресеты](../configs/v4) можно копировать и менять.
Старые конфиги/архивы schema 3 не подходят.

### Установка и конфиг

**Нужна GPU:** setup ставит lock в отдельный venv и проверяет CUDA.
На уже подготовленной машине повторять setup не нужно.

```bash
python3.13 tools/benchmark.py setup --profile team-rtx --environment ./u1-env
cp configs/v4/koala.json configs/my-koala.json
cp examples/u1_v4_speed_import.json configs/my-koala-data.json
./u1-env/venv/bin/python tools/benchmark.py auth-check --config configs/my-koala.json
```

Используется существующий HF-кеш авторизации либо `HF_TOKEN`/`HF_TOKEN_PATH`.
Не вставляйте токен в команды, JSON или архив. `auth-check` не загружает веса.
В `configs/my-koala-data.json` замените `configs[0].config` на
`configs/my-koala.json`, `name` на `my-koala`, `version` на `my-koala-v1`.
Остальные поля этого примера оставьте без изменений.

### Данные и план

**Без GPU и весов:** загрузка корпуса и CPU-подготовка tokenizer/SDK.
Если исходный корпус уже есть, пропустите fetch-data. Каталог выхода prepare-data
должен быть новым.

```bash
./u1-env/venv/bin/python tools/benchmark.py fetch-data --output data/speed/role-corpus
./u1-env/venv/bin/python tools/benchmark.py prepare-data --spec configs/my-koala-data.json --output data/speed/my-koala-v1
```

Продолжайте только при `status=ready`. Из JSON-вывода prepare-data перенесите
`path` и `manifest_sha256` в `data.path` и `data.manifest_sha256` файла
`configs/my-koala.json`. В этом же конфиге можно изменить
`batch_sizes_by_bucket`; по умолчанию Koala B1=[1,256]. Затем:

```bash
./u1-env/venv/bin/python tools/benchmark.py plan --config configs/my-koala.json --environment ./u1-env
```

Проверьте `data_ready`, `preflight_workers`, `latency_workers`,
`timed_batches` и `timed_input_occurrences`. `plan` не проверяет GPU.
Исходные сообщения и индекс будут в каталоге данных; manifest закрепляет их хеши.

### Запуск baseline

**GPU:** выберите при необходимости один короткий маршрут проверки. Preflight
не измеряет основные серии; pilot добавляет одну короткую серию. Полный baseline
сам выполняет preflight, отдельные проверки не являются обязательным дублем.

```bash
./u1-env/venv/bin/python tools/benchmark.py preflight --config configs/my-koala.json --environment ./u1-env --output runs/my-koala-smoke
./u1-env/venv/bin/python tools/benchmark.py run --config configs/my-koala.json --environment ./u1-env --pilot --output runs/my-koala-pilot
```

**GPU, полный прогон** для reference:

```bash
./u1-env/venv/bin/python tools/benchmark.py run --config configs/my-koala.json --environment ./u1-env --output runs/my-koala-baseline
```

Полный reference должен иметь `baseline_complete`; smoke/pilot его не заменяют.
Каждому запуску нужен новый output: существующие каталоги и ZIP не перезаписываются.

### Когда готов candidate

Создайте `configs/my-koala-candidate.json` из своей baseline-конфигурации:
сохраните baseline, данные и сетку, добавьте **полный** `candidate` и задайте
`mode="search"` и соответствующий `comparison`. Candidate не наследует
поля baseline. Используйте поддерживаемый адаптер; [внешний адаптер](#6-свой-candidate-и-quality)
нужен, если реализация другая. Для готового примера compile/ORT используйте
GLiNER-пресеты из [раздела запусков](#4-baseline--поиск--подтверждение), не переносите
GLiNER-runtime на Koala.

**GPU:** search запускает только candidate; confirm заново измеряет обе стороны.

```bash
./u1-env/venv/bin/python tools/benchmark.py run --config configs/my-koala-candidate.json --environment ./u1-env --baseline-ref runs/my-koala-baseline --output runs/my-koala-search
./u1-env/venv/bin/python tools/benchmark.py run --config configs/my-koala-candidate.json --environment ./u1-env --mode confirm --output runs/my-koala-confirm
```

Изменённая сетка требует совместимого нового baseline; reference не пересчитывается
скрыто. Confirm не использует старые timing. Для положительного confirm требуется
также валидное свидетельство [проверки мониторинга](#5-память-профили-и-статусы)
на соответствующей среде; без него результат не получает confirm_pass.

**Offline, без GPU:**

```bash
python3.13 tools/benchmark.py report --output runs/my-koala-search
python3.13 tools/benchmark.py table --inputs runs/my-koala-baseline runs/my-koala-search --output runs/my-koala-table.json
```

В каждом запуске: `status.json`, `resolved_run.json`, `summary.json/.md`,
worker-evidence и архив `<output>.zip`. Для общей таблицы передайте архив, а не
только скриншот. [Quality](#6-свой-candidate-и-quality) использует отдельный набор
и `quality_batch_sizes`, без speed-бакетов.

**Особый случай Open:** сначала получите точный task2510-шаблон/готовые данные
у У1, как описано [ниже](#воспроизведение-восьми-прежних-пресетов-дополнительный-маршрут).
Исходный HF-шаблон недоступен; чистый клон сам по себе этот артефакт не содержит.
Не заменяйте его native0124-шаблоном.

## 1. Среда и доступ

Linux x86_64 / Python 3.13, только GPU 0. RTX 4090, RTX 5090 и T4 не смешиваются.
Установка в новый venv, без обновления host PyTorch или автоматической смены версий:

```bash
python3.13 tools/benchmark.py setup --profile team-rtx --environment ./u1-env
./u1-env/venv/bin/python tools/benchmark.py auth-check --config configs/v4/shield.json
```

Прежний [RTX-lock](../env/team_rtx.lock.json) не изменён. Профиль `team-t4-laya`
сохраняет T4 и тот же расширенный стек; `team-t4` остаётся прежним окружением.
Setup проверяет пакеты и CUDA, не подтверждает inference всех моделей.
HF использует штатный кеш авторизации, `HF_TOKEN` или `HF_TOKEN_PATH`.
Токен не добавлять в JSON, командную строку, репозиторий или архив.

## 2. Данные

### Общий корпус и готовые индексы

Корпус сообщений общий, а индексы token IDs/масок/длин и списки ID бакетов
зависят от нативной подготовки. Названия восьми моделей в HF обозначают готовый
предрасчёт, а не ограничение на другие модели или отдельные тексты под них.
`run` не собирает отсутствующие бакеты автоматически: он читает закреплённые
списки, а preflight сверяет фактические входы с индексом. Токенизация внутри
измеряемого запроса всё равно выполняется, готовые GPU-входы не кешируются.

В v4 индекс выбирается по `preparation_identity`: хешу `adapter`, `tokenizer`,
`sdk_assets`, `template`, `preparation`. Привязки к имени весов в этом ключе нет.

### Воспроизведение восьми прежних пресетов: дополнительный маршрут

Опубликованный [ролевой корпус](https://huggingface.co/datasets/vXofi/guardrails-speed-en-ru/tree/7324b8823ded971c05076e0c7f4ace5d9b867863)
не меняется. Для формата измерителя v4 требуется новый локальный индекс:

```bash
./u1-env/venv/bin/python tools/benchmark.py fetch-data --output data/speed/role-corpus
./u1-env/venv/bin/python tools/benchmark.py prepare-data --spec configs/v4/reindex.json --output data/speed/role-corpus-v4-20261112
```

Для Open в `reindex.json` отдельно указан проверенный task2510-overlay:
`data/speed/role-corpus-open-task-v2`, manifest
`da255a70164f3175da8c29c313c07ea8ce1d58a7a4a2993ea692f2591bbf431c`.
Он нужен для проверки паритета с прежним task-контрактом, а не с native-индексом HF.
Если этого локального артефакта нет, получить его у У1 вместе с готовой локальной
сборкой данных. Непроверенный индекс или native-шаблон не подставляются.
HF-репозиторий шаблона 2510 сейчас отвечает 404; точный закреплённый файл доступен
в сохранённом кеше. Для чистой машины он также необходим, с проверкой ревизии/хеша.

Пересборка выполняется только на CPU, без весов. У всех восьми пресетов сравниваются
token IDs, masks и длины всех 12000 записей; прежние списки ID сохраняются только
при совпадении. Две локальные сборки дали одинаковый manifest:
`23785e17e3ef29b302da253189fadb7c5ed17691155e7205faa5ac508641a74b`.
В каждом из 20 модельных бакетов остаётся 400 эффективных уникальных входов.
Имя «v4» опубликованного корпуса не означает совместимость его старого manifest
с новой схемой локальных данных.

Пути и `data.manifest_sha256` в пресетах должны указывать на эту сборку.
`reindex` проверяет паритет со старыми индексами и сохраняет прежние списки ID.
Для новой модели или изменённой подготовки этот режим не использовать:
нужен обычный speed-import ниже, без исторического Open-overlay.

### Добавить новую модель на общем корпусе

Все команды выполняются из корня репозитория. Нужны общий `corpus.jsonl`,
CPU-зависимости выбранного preparer и доступ к закреплённым tokenizer/SDK-assets.
Подготовка не загружает веса и не использует GPU; установка `setup` и последующий
`preflight` имеют собственные требования к оборудованию.

1. Создать полный baseline-конфиг, например `configs/my-model.json`, на основе
   подходящего пресета. Указать свои веса, tokenizer/assets, template, правила
   подготовки/scoring, лимит и `buckets`. Совпадения архитектуры недостаточно:
   встроенный адаптер должен поддерживать именно этот вход и смысл ответа.
   Иначе подключить [внешний адаптер с CPU-preparer](#6-свой-candidate-и-quality).
2. Взять [готовый speed-import JSON](../examples/u1_v4_speed_import.json).
   Без изменений он строит индекс Koala на общем корпусе. Для своей модели
   скопировать его в `configs/my-model-data.json`, заменить `configs[0].config`
   на `configs/my-model.json`, `name` на свой идентификатор и `version` на версию
   сборки. `name` служит подписью; identity вычисляется из подготовки.
   `source` оставить тем же: новых текстов под модель создавать не нужно.
3. Выполнить сборку в **новый**, ещё не существующий каталог:

```bash
./u1-env/venv/bin/python tools/benchmark.py fetch-data --output data/speed/role-corpus
./u1-env/venv/bin/python tools/benchmark.py prepare-data --spec configs/my-model-data.json --output data/speed/my-model-v1
```

Если общий корпус уже скачан, `fetch-data` пропустить. Пути внутри JSON разрешаются
от текущего рабочего каталога, не от расположения JSON. При `prepare-data` из
модельного конфига используются выбранный вариант и `buckets`; указанный в нём
`data` не читается, до завершения сборки можно оставить значения исходного пресета.

Сборка проверяет паритет нативной подготовки, индексирует все сообщения и выбирает
по 400 уникальных эффективных входов на обязательный бакет. Бакет определяется
полной длиной входа до padding. Восемь интервалов длины стремятся набрать по 50
примеров; дефицит перераспределяется, без padding ради попадания в бакет.
Одинаковые подготовленные входы не считаются разными из-за игнорируемого контекста.

На выходе:

```text
corpus.jsonl
indices/<preparation_identity>.jsonl
buckets/<preparation_identity>/B1.json
manifest.json
```

Остальные бакеты появляются согласно конфигу. Сообщения, ID и происхождение
сохраняются; копия общего корпуса в этом каталоге не становится отдельной
генерацией текстов. HF-публикация не требуется и этой командой не меняется.

4. Проверить `status` в выводе и `manifest.json`. `ready` требует 400 входов
   в каждом обязательном бакете и отсутствия blockers. В manifest смотреть
   `blockers`, `indices` и `model_buckets`: подпись подготовки, CPU-паритет,
   `selected`, `eligible_unique_inputs`, `histogram_8`. Фактические полные длины
   находятся в соответствующем `indices/<preparation_identity>.jsonl`.
   `not_ready` не означает, что набор можно использовать для полного baseline:
   при дефиците нужны дополнительные общие сообщения, при ошибке preparer надо
   устранить её причину. Небольшие непустые бакеты допустимы только для smoke/pilot,
   если для них уже есть корректный индекс.
5. Из результата `prepare-data` перенести `path` в `data.path`,
   `manifest_sha256` в `data.manifest_sha256` модельного конфига, затем проверить:

```bash
./u1-env/venv/bin/python tools/benchmark.py plan --config configs/my-model.json --environment ./u1-env
./u1-env/venv/bin/python tools/benchmark.py preflight --config configs/my-model.json --environment ./u1-env --output runs/my-model-smoke
```

`plan` не использует GPU, но `preflight` уже загружает модель на GPU и проверяет
реальный путь. `ready` у данных или `plan` не подтверждает GPU-совместимость.
Baseline и candidate одной модели получают этот же закреплённый набор;
candidate не пересобирает себе бакеты. Далее использовать
[обычный baseline/search/confirm](#4-baseline--поиск--подтверждение).

### Когда повторять подготовку

| Изменение | Что делать с данными |
| --- | --- |
| Только веса, dtype, runtime/attention backend или batch sizes при той же подготовке | Переиспользовать индекс; совместимость исполнения и ограничения batch проверить отдельно |
| Tokenizer/revision, SDK-assets, шаблон, роли, special tokens, лимит или adapter (включая хеш внешнего Python-модуля) | Повторить `prepare-data`: identity подготовки изменится |
| Границы бакетов, тексты, роли сообщений или состав корпуса | Повторить `prepare-data`; прежние списки ID и manifest больше не подходят |

Для своих сообщений в том же speed-import поменять `source`, отображение `fields`,
`version` и `provenance`. Обязательны `id` и ровно одно из `messages`/`text`;
`text` импортируется как одно сообщение user. Отображённые поля должны присутствовать
в каждой строке. Сохранить атрибуцию источников. Для нескольких подготовок добавить
записи `{config, variant, name}` в `configs`, не делать отдельные исходные тексты
под каждую модель. Для engineering/parameter/quantization с неизменной подготовкой
достаточно baseline-индекса; смена адаптера candidate может требовать новой проверки
подготовки, а не своего независимого отбора ID.

## 3. Конфиг и план

[Восемь baseline-конфигов](../configs/v4) задают полные независимые варианты:
`adapter, weights, tokenizer, template, preparation, scoring, execution`.
Candidate не наследует незаполненные поля baseline. HF-источники закреплены
полными revision; локальные файлы требуют SHA256, каталоги — полного inventory.
Неподдержанные настройки отклоняются, а не игнорируются.

Адаптер выбирается явно, не по имени checkpoint. Совместимые другие веса задаются
в конфиге; новая постановка задачи/парсер требуют подходящего адаптера.
Runtime eager/compile/ORT, full attention и linear attention разделены.
Laya сохраняет native SDK AMP; смена precision между workers запрещена.

### Входы моделей

Ниже роли именно включённых пресетов; другой контракт требует своей подготовки
и индекса, а не скрытой подмены сообщения.

| Пресет | Оцениваемый вход |
| --- | --- |
| GLiNER, Koala, Llama, Laya | Assistant, если он есть; иначе user. Игнорируемый user не склеивается с ответом |
| HaloGuard | Последний user с авторским системным промптом и Safety cue; assistant-only неприменим |
| HiveTraceGuard-Pro | Последнее сообщение через собственный шаблон; ответ в паре с контекстом user |
| ShieldGemma | User-only либо user→assistant, соответствующая Dangerous Content формулировка; assistant-only неприменим |
| OpenGuardrails | User-only, веса/tokenizer 0124 mirror, закреплённый task-шаблон 2510, greedy до 10 токенов |

Неприменимые форматы остаются в coverage. Нативная подготовка не означает
подтверждённое качество модели на каждом формате.

```bash
./u1-env/venv/bin/python tools/benchmark.py plan --config configs/v4/gliner.json --environment ./u1-env
```

`plan` читает индексы/reference, но не загружает веса и не использует GPU.
Показывает timed batches, число обработок с повторами, workers и прогревы.
Backend-подготовка зависит от форм/runtime; её время сохраняется после запуска.
Границы бакетов задаются конфигом, до padding:
B1 1–512, B2 513–1024, B3 1025–2048, B4 2049–4096.

### Статические батчи

| Класс | Бакет | Рабочая сетка |
| --- | --- | --- |
| Encoder | B1 | 1, 256 |
| Decoder | B1 | 1, 64 |
| Decoder | B2 | 1, 32 |
| Decoder | B3 | 1, 16 |
| Decoder | B4 | 1, 8 |

Это конфигурация, не утверждение о вместимости всех моделей. Допустим любой
положительный целый B ≤ N; без сетки используется B=1. OOM/unsupported сохраняются
по ячейкам, остальные продолжаются. Размер, dtype и устройство не меняются автоматически.

Сетка разрежена до 2 ячеек на encoder и 8 на decoder (раньше 4 и 15).
Меняется только JSON-пресет, не поддержка промежуточных размеров. Минимум
100 батчей в основной серии остаётся: при B256 это 25600 обработок 400 входов.
Больший batch не сокращает такой прогон автоматически. Реальное wall-clock
ускорение инструмента не измерялось; экономятся ячейки, повтор baseline в search
и отдельные memory-проходы. Reference должен иметь ту же сетку.

## 4. Baseline → поиск → подтверждение

```bash
./u1-env/venv/bin/python tools/benchmark.py preflight --config configs/v4/gliner.json --environment ./u1-env --output runs/gliner-smoke
./u1-env/venv/bin/python tools/benchmark.py run --config configs/v4/gliner.json --environment ./u1-env --pilot --output runs/gliner-pilot
./u1-env/venv/bin/python tools/benchmark.py run --config configs/v4/gliner.json --environment ./u1-env --output runs/gliner-baseline
./u1-env/venv/bin/python tools/benchmark.py run --config configs/v4/gliner_compile.json --environment ./u1-env --baseline-ref runs/gliner-baseline --output runs/gliner-compile-search
./u1-env/venv/bin/python tools/benchmark.py run --config configs/v4/gliner_compile.json --environment ./u1-env --mode confirm --output runs/gliner-compile-confirm
```

Для FLA/ORT аналогично: `halo_fla.json` и `gliner_ort.json`.
`--baseline-ref` принимает каталог или ZIP полного baseline. В confirm этот
параметр запрещён. По умолчанию без candidate выбран baseline, с candidate — search.

| Режим | Что выполняется |
| --- | --- |
| preflight / smoke | Проверка фактических входов, native batching, backend, A→B→A, границ; без основных серий |
| baseline | Preflight; A/A две пары ×30; три основные серии baseline |
| search | Только candidate: preflight и три основные серии по сохранённым планам reference; baseline/A/A не повторяются |
| confirm | Новый preflight обеих сторон; A/A две пары ×30; три A/B-пары с чередованием AB/BA |
| pilot | Preflight и одна основная серия/пара ×12; без A/A и без итогового pass |
| quality | Отдельные данные и сетка; один проход с неполным хвостом; без скоростных серий |

Перед каждой timed-стороной/серией 20 прогревочных батчей. Каждая сторона — свежий
процесс. Компиляция/экспорт/необходимые формы готовятся вне latency; eager не делает
дополнительный полный проход. Backend-proof выполняется в preflight, не заново в
каждой серии. Основная серия: `max(100, ceil(N/B))` полных батчей.

Search сверяет baseline-конфиг, входы, планы, код, протокол, GPU/VRAM, драйвер,
CUDA, пакеты, вычислительные настройки и метод памяти. Несовместимость — ошибка,
а не повод автоматически запускать baseline. Совпавшие параметры не доказывают
отсутствие дрейфа между сессиями: search даёт только предварительный результат,
без парных доверительных интервалов. Полный confirm нужен для итогового вывода.

## 5. Память, профили и статусы

Память снимается одновременно со скоростью. Один процесс nvidia-smi на worker,
запрошенный интервал 100 мс; setup/inference разделены. Выборочный пик процесса
может пропустить короткие всплески. Torch allocated/reserved и размер артефакта
показаны отдельно; Torch не заменяет память ORT. Ошибка sampler не скрывает latency.
Трёх дополнительных memory-проходов больше нет. `--profile-trace` включает отдельные
short/long-профили; без флага они не запускаются.

Однократная техническая проверка sampler:

```bash
./u1-env/venv/bin/python tools/check_v4_monitoring.py --configs configs/v4/gliner.json configs/v4/open.json --environment ./u1-env --output runs/monitoring-check
```

Её `monitor_validation.json` можно указать в конфиге через `monitor_validation`.
До успешной проверки на той же среде confirm не получает положительный итог.
Это не дополнительные тесты перед каждым candidate.

- `baseline_complete`: комплектный reference, не ускорение.
- `exploratory`: предварительный поиск относительно сохранённого reference.
- `confirm_pass`: парное подтверждение в одной сессии, не независимая репликация.
- `pilot_complete / preflight_passed`: короткая проверка, не benchmark.
- `inconclusive / partial / incomplete`: шум, недостающие стадии или блокер; детали по ячейкам.
- `quality_complete`: оценка закончена; не означает «качество сохранено».

Инженерные изменения: строго >20% снижения p95, параметрические >10%, плюс прежние
проверки шума и вердиктов. «Быстрее во всей сетке» отдельно от достижения порога.
Квантование не получает автоматического quality-approval.

## 6. Свой candidate и quality

[Внешний адаптер-пример](../examples/u1_v4_external_adapter.py) делегирует GLiNER:
это пример интерфейса, не квантование. [Конфиг](../examples/u1_v4_external.json)
закрепляет Python-файл и зависимости. Изменение кода требует нового хеша.
Модуль предоставляет `create_preparer(spec)` и `create(spec,directory)`.
Preparer: `index, parity, close`, без весов/GPU.
Runtime: `index, request_batch, verify_batch, prepare_batch_backend,
prove_batch_backend, reject_batch_over_limit, boundary, size, close`,
`metadata.precision`, `batch_policy` и `compile_count`.
Batch-запрос возвращает упорядоченные native-ответы; цикл single не является batching.
Для внешнего адаптера фиксированный словарь вердиктов можно явно задать в его
`scoring.labels`; одних runtime `metadata.labels` недостаточно для исключений
при защите закрытого содержимого. Возврат исходного текста вместо класса отклоняется.
Каждая строка сверяется с индексом. Инженерный candidate сохраняет веса и precision;
для квантованного артефакта нужен `quantization.baseline_identity` точного baseline.

Quality не читает speed-корпус и не делится автоматически на бакеты:

```bash
./u1-env/venv/bin/python tools/benchmark.py prepare-data --spec examples/u1_v4_quality_import.json --output data/team-quality
./u1-env/venv/bin/python tools/benchmark.py quality --config examples/u1_v4_quality.json --environment ./u1-env --output runs/gliner-quality
```

В примерах заменить путь набора, manifest SHA256, mapping и при необходимости модель.
Строка содержит messages (или text), target, target_role/target_index, slices.
Метка ответа не оценивается как метка запроса. `quality_batch_sizes` независимо
задаёт размеры, по умолчанию [1]; хвост не отбрасывается, записей не повторяют.
Baseline/candidate получают одинаковый порядок; сравнение по общим успешным ID
с явным знаменателем. Single→batch доступно при наличии B=1.
Метрики: confusion matrix, accuracy, macro-F1, при явно бинарной задаче FPR/FNR,
срезы, coverage, неопределённые знаменатели. Требования качества задаёт У2.
Сообщения, промпты и private-логи не входят в публичный архив.

## 7. Результаты и проверка

```bash
python tools/benchmark.py report --output runs/gliner-compile-search
python tools/benchmark.py table --inputs runs/gliner-baseline runs/gliner-compile-search --output runs/table.json
```

`resolved_run.json` неизменяем: эффективный конфиг, ID/индексы, планы, версии и
среда. Inline-промпты в публичном снимке заменены устойчивыми SHA256-идентичностями;
исполняемый конфиг хранится только в `.private`, проверяется по хешу и не архивируется.
Не встраивать секреты или закрытые тексты в Python-исходники адаптера: исходники
включаются в архив для воспроизводимости. Workers пишут отдельные evidence; search включает необходимую копию
baseline. Offline-report не обращается к моделям, исходному корпусу или HF и не
выполняет код из архива. Каталог/ZIP не перезаписываются; ZIP называется полным
именем каталога плюс `.zip`. При ошибке остаются завершённые стадии и безопасный
частичный архив. Общий лимит сравнения — 10 часов.

CPU-тесты и повторная индексация не подтверждают RTX-совместимость.
[Проверка публикуемой редакции](../results/u1_handoff/2026-11-12-v4-release/validation.json):
490 CPU-тестов прошли в чистой копии без ошибок и пропусков.
Историческая [повторная индексация](../results/u1_handoff/2026-11-12-v4-cpu/validation.json)
подтвердила воспроизводимость индексов; корпус этим этапом не менялся.
[Верхние smoke-конфиги](../configs/v4/smoke) охватывают четыре encoder B1×256
и Open B1×64/B2×32/B3×16/B4×8.
[Новый Open BF16/native](../results/u1_handoff/2026-11-12-v4-native-open/README.md)
прошёл все четыре верхние ячейки. Остальные новые режимы этим не проверены.
Полные baseline/search/confirm GPU-серии в текущую приёмку не входят.
