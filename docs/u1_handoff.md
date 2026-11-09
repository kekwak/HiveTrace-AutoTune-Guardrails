# У1: запуск общего измерителя

Актуально на **9 ноября 2026**. Это основная операционная инструкция для команды:
рабочая ветка `results/haloguard-gliner-u1`, код, конфигурации и lock в GitHub.
Для обычной работы ZIP и новый notebook не нужны.

[Протокол](measurement_protocol_v1.md) · [режимы baseline](baseline_modes.md) ·
[матрица готовности](u1_handoff_readiness.md) · [план У1](u1_protocol_plan.md) ·
[тезисы для синка](u1_sync_brief.md).

**Проверка 4 ноября:** 151 CPU-тест без ошибок/пропусков и отдельный smoke **8/8 на T4**:
40 таймингов полного запроса, входы/роли, A→B→A, границы, CUDA backend и память.
[Validation](../results/u1_handoff/2026-11-04-review-smoke/validation.json),
[исходники запуска](../results/u1_handoff/2026-11-04-review-smoke/source_manifest.json).
Это не screening, проверка RTX или оценка качества. Lock повторно установлен на T4;
RTX 4090/5090 проверяют владельцы. Исторические approvals не переносятся.

**Исправление Laya, 7 ноября:** ошибка `Laya native FP16 autocast changed` на 4090
возникала из-за T4-зависимого ожидания в адаптере. SDK читает `amp_dtype=bf16` из
checkpoint на новых GPU, тогда как T4 использует FP16. Адаптер сохраняет нативный
выбор SDK без изменения FP32-весов или конфига. `metadata.precision` содержит dtype
параметров, включённость/dtype autocast и политику `native_sdk`.
CPU fallback, отключённый AMP или последующая смена режима останавливают worker,
в том числе в quality. Исправление прошло отдельный
[smoke Laya на RTX 4090](../results/u1_handoff/2026-11-08-laya-rtx4090/validation.json):
FP32-веса + нативный BF16 autocast, точный 97-wheel lock, текущие исходники.
9 ноября encoder batch-smoke на RTX 5090 прошёл для GLiNER, Koala, Llama и Laya:
[20/20 ячеек B1, batch 1/8/16/32/64](../results/u1_handoff/2026-11-09-encoder-batch-rtx5090/validation.json).
Отдельно прошёл [HaloGuard batch-smoke: 11/11 ячеек B1–B4](../results/u1_handoff/2026-11-09-halo-batch-rtx5090/validation.json).
HiveTraceGuard-Pro, OpenGuardrails, ShieldGemma и полный screening на RTX не проверены.
T4-smoke 4 ноября относится к прежней версии исходников,
не к этому патчу; lock, корпус и индексы не менялись.
Контракт подготовки сравнивается отдельно от вычислительных `rules`, поэтому смена
autocast не требует переиндексации Laya. Полный inference-контракт workers проверяется
строго; CPU-проверки не заменяют GPU-smoke.

## 1. Установка и авторизация

Команды из корня клона рабочей ветки. Нужны Linux x86_64, Python 3.13 и GPU 0 —
RTX 4090 либо RTX 5090. Установочный каталог должен быть новым:

```bash
python3.13 tools/benchmark.py setup --profile team-rtx --environment ./u1-rtx-env
./u1-rtx-env/venv/bin/python tools/benchmark.py auth-check --model shield
```

[Lock](../env/team_rtx.lock.json) содержит 97 точных wheels и SHA256: прежний стек
плюс `laya==0.3.20` без extras. PyTorch CUDA 13 и CUDA 12 библиотеки ORT сосуществуют.
Установка идёт в отдельный venv, host Python/PyTorch не меняются. Никаких автоматических
обновлений или CPU-fallback. Не добавлять случайные зависимости из-за известного
host-предупреждения `sitecustomize/wrapt`; при ошибке сохранить логи и конкретный blocker.
Успешный setup подтверждает inventory/CUDA, не inference всех моделей на RTX.

HF-доступ предоставляет владелец runtime: существующий login/cache либо
`HF_TOKEN` / `HF_TOKEN_PATH`, заданные до запуска. `auth-check` читает только
метаданные, не скачивает веса. JSON показывает наличие credentials и класс ошибки,
но не токен. Нужен аккаунт с разрешённым доступом ShieldGemma; токен не передаётся
аргументом, в конфигурации или архиве.

Для отдельного короткого smoke на Kaggle T4 существует явный профиль
`team-t4-laya` с теми же 97 wheels, но проверкой T4 на GPU 0.
Старый `team-t4` сохраняет 96-wheel lock и не является текущей средой всех восьми моделей.
В приватном Kaggle launcher разрешить Secret `HF_TOKEN` только этому notebook;
значение из `UserSecretsClient` передать процессу через окружение и не печатать.

## 2. Получение данных и Open-индекс

Сначала получить закреплённый исходный корпус:

```bash
./u1-rtx-env/venv/bin/python tools/benchmark.py fetch-data
```

[HF v4](https://huggingface.co/datasets/vXofi/guardrails-speed-en-ru/tree/7324b8823ded971c05076e0c7f4ace5d9b867863)
закреплён в [профиле данных](../configs/data/role_speed_v4.json).
Commit: `7324b8823ded971c05076e0c7f4ace5d9b867863`.
Manifest: `760adad3e5946384de9fec98aebd9327229cc02c706af094322f2f37e95fc02c`.
Каталог назначения: `data/speed/role-corpus`; существующий каталог не перезаписывается.
`--source` допускает локальную копию с тем же manifest; непроверенный `main` не подставляется.

**До запуска OpenGuardrails** пересобрать только его индекс на CPU, без весов:

```bash
./u1-rtx-env/venv/bin/python tools/reindex_role_model.py \
  --source data/speed/role-corpus --output data/speed/role-corpus-open-task-v2 \
  --models open --version u1-role-speed-v4-open-task2510-v2
```

Ожидаемый [производный профиль](../configs/data/role_speed_open_task.json):
manifest `da255a70164f3175da8c29c313c07ea8ce1d58a7a4a2993ea692f2591bbf431c`.
Общие 12000 сообщений и файлы семи остальных моделей сохраняются байтово.
Open-пресет уже ссылается на производный каталог; копировать туда старый индекс нельзя.
`fetch-data` проверяет целостность опубликованного v4 и показывает для Open
`requires_reindex`: это ожидаемое отличие контракта, не повреждение загрузки.
Новый Open snapshot на HF пока не публиковался; пересборка воспроизводит manifest.
Производный профиль v2 фиксирует исправленный сборщик, который не требует пересборки
другой модели из-за одних вычислительных правил. Тексты, token IDs/маски и бакеты
совпадают с v1; опубликованный HF-корпус не менялся. Старый v1 не перезаписывается
и остаётся применимым при явном указании его прежних path/manifest в конфигурации.

### Входы моделей

| Модель | Исходные сообщения и цель оценки |
| --- | --- |
| GLiNER / Koala / Llama / Laya | Assistant, если есть; иначе user. Только выбранный текст, затем нативная подготовка/SDK |
| HaloGuard | Последний user, авторский prompt, thinking off и `Safety: `. В паре ответ не используется; assistant-only неприменим |
| HiveTraceGuard-Pro | Собственный шаблон; цель — последнее сообщение, предшествующий user остаётся контекстом. Assistant-only принимается технически, качество без контекста не подтверждено |
| ShieldGemma | User-only или user+assistant; соответственно оценка запроса либо ответа с разными авторскими Dangerous Content guidelines. Assistant-only неприменим |
| OpenGuardrails | Только user-only. Веса/tokenizer 0124, закреплённый task-шаблон 2510; greedy до 10 токенов, native EOS, строгий parser |

Open использует профиль `open-0124-task-2510-v1`: родной шаблон 0124 воспроизводимо
повторял метки, смена precision/attention/cache это не устранила. По отдельному
согласованию возвращён task-промпт 2510 с теми же весами; parser не ослаблен.
[Диагностика](../results/u1_handoff/2026-11-01-open-shield/diagnostics.json),
[проверка входов](../results/u1_handoff/2026-11-01-open-shield/input_validation.json).
Исторические результаты native-профиля не переписываются.

### Корпус и модельные бакеты

Запись: `id`, `messages: [{role, content}]`, `provenance`, `content_sha256`.
Импорт старого `text` означает один user. Изменение ролей/содержимого создаёт новую
идентичность; синтетически назначенная роль не выдаётся за настоящий ответ LLM.

Полная длина включает prompt/schema, chat template и специальные токены:
B1 1–512, B2 513–1024, B3 1025–2048, B4 2049–4096.
Энкодерам обязателен B1; четырём декодерам — B1–B4.
На модельный бакет — 400 уникальных эффективных входов, суммарно по применимым форматам.
Бакеты выбираются независимо для модели; пересечение списков ID между моделями
не требуется. Различия проигнорированного контекста не создают новый эффективный вход.

Отбор стремится к 50 входам в каждом из восьми интервалов длины, перераспределяя
дефицит без padding/усечки. Пробелы и состав видны в
[отчёте корпуса](https://huggingface.co/datasets/vXofi/guardrails-speed-en-ru/blob/7324b8823ded971c05076e0c7f4ace5d9b867863/REPORT.md).
Это нагрузка скорости, а не размеченный safety-бенчмарк или production-выборка.

## 3. Preflight, pilot и полный запуск

В [configs/benchmark-v3](../configs/benchmark-v3) восемь baseline-пресетов и три сравнения:
`halo_fla.json`, `gliner_compile.json`, `gliner_ort.json`.
Первым запускайте baseline своей модели. Пример GLiNER:

```bash
./u1-rtx-env/venv/bin/python tools/benchmark.py preflight --config configs/benchmark-v3/gliner_baseline.json --environment ./u1-rtx-env --output runs/gliner-preflight-1
./u1-rtx-env/venv/bin/python tools/benchmark.py run --pilot --config configs/benchmark-v3/gliner_baseline.json --environment ./u1-rtx-env --output runs/gliner-baseline-pilot-1
```

Preflight проверяет все 400 выбранных входов каждого обязательного бакета, фактические
token IDs/маски, конечность, повторяемость, A→B→A, лимит/превышение до forward и backend.
Он не перебирает весь исходный пул из 12000 сообщений и не запускает latency-серии.
`run --pilot` повторяет preflight, оставляет A/A две пары ×30 и одну основную серию
×12 на бакет; не получает pass ускорения и не включает полные memory/profile-стадии.

Для короткого smoke в отдельной копии пресета поставить `"mode": "smoke"` и вызвать
обычный `run`. Проверяются два входа каждого бакета, CUDA, границы и память;
лимит час на модель. Это не полный baseline. Smoke не заменяет preflight полного набора.

После успешной проверки владелец **явно** запускает полный baseline:

```bash
./u1-rtx-env/venv/bin/python tools/benchmark.py run --config configs/benchmark-v3/gliner_baseline.json --environment ./u1-rtx-env --output runs/gliner-baseline-1
```

Без `candidate`: A/A две пары ×30 и три серии ×400 на бакет,
три отдельных memory-процесса и short/long-профили. Нет фиктивной второй модели.
Полные запуски не выполняются автоматически вслед за pilot.

Сравнение готовой конфигурации — отдельный новый каталог и собственный контроль:

```bash
./u1-rtx-env/venv/bin/python tools/benchmark.py preflight --config configs/benchmark-v3/gliner_compile.json --environment ./u1-rtx-env --output runs/gliner-compile-preflight-1
./u1-rtx-env/venv/bin/python tools/benchmark.py run --pilot --config configs/benchmark-v3/gliner_compile.json --environment ./u1-rtx-env --output runs/gliner-compile-pilot-1
./u1-rtx-env/venv/bin/python tools/benchmark.py run --config configs/benchmark-v3/gliner_compile.json --environment ./u1-rtx-env --output runs/gliner-compile-1
```

Не переходить к полному запуску при неуспешном preflight/pilot.
Параметры A/A/A-B, p95, шум, память и критерии находятся только в
[протоколе](measurement_protocol_v1.md), а не переопределяются примером команды.
Halo FLA — доступный исследовательский кандидат, **не рекомендация включать по умолчанию**:
в позднем полном [T4-screening](benchmark.md#haloguard-v4-полный-отрицательный-результат)
он замедлил запросы.

### Статические батчи

**Batch sizes пока не финальные.** Пресеты задают исследовательскую сетку, а не
гарантированно подходящие размеры для каждой модели. До полного запуска на своей
модели/backend выполнить smoke; отсутствие OOM на другой модели не является проверкой.
Изменять сетку можно до запуска с сохранением конфигурации, но не автоматически
после OOM внутри измерений.

Новые пресеты лежат в [configs/benchmark-batch](../configs/benchmark-batch), отдельные
encoder-smoke в [configs/smoke-batch](../configs/smoke-batch). Одиночные пресеты выше
не изменены. Поле `batch_sizes_by_bucket` задаёт всю матрицу до старта, например
`{"B1": [1, 8, 16, 32, 64]}` для encoder; все обязательные бакеты модели должны
присутствовать. Меньшая явно заданная сетка допустима, но её результаты не считаются
прохождением более широкой сетки. Полный протокол и объём повторов описаны в
[разделе батчей](measurement_protocol_v1.md#статические-батчи).

После установки прежнего RTX-lock и получения того же корпуса:

```bash
./u1-rtx-env/venv/bin/python tools/benchmark.py run --config configs/smoke-batch/gliner.json --environment ./u1-rtx-env --output runs/gliner-batch-smoke-1
./u1-rtx-env/venv/bin/python tools/benchmark.py run --pilot --config configs/benchmark-batch/gliner_compile.json --environment ./u1-rtx-env --output runs/gliner-batch-pilot-1
./u1-rtx-env/venv/bin/python tools/benchmark.py run --config configs/benchmark-batch/gliner_baseline.json --environment ./u1-rtx-env --output runs/gliner-batch-baseline-1
./u1-rtx-env/venv/bin/python tools/benchmark.py run --config configs/benchmark-batch/gliner_compile.json --environment ./u1-rtx-env --output runs/gliner-batch-compile-1
```

Это последовательность отдельных решений владельца, не команда автоматического
запуска всех этапов. **Encoder baseline-smoke прошёл на RTX 5090** для `gliner`,
`koala`, `llama`, `laya`, вся сетка 1/8/16/32/64. Дополнительно `halo` прошёл
всю decoder-сетку B1–B4 (11 ячеек), пресет `configs/smoke-batch/halo.json`.
Это не screening: батчевые compile/ORT/FLA, остальные decoder и скорость требуют
отдельных запусков. Batch 1 также измеряется
заново на целевой карте. [Доказательства и ограничения](u1_handoff_readiness.md#статические-батчи-9-ноября-2026).

В [configs/smoke-upper](../configs/smoke-upper) отдельно сохранены экспериментальные
точки OpenGuardrails и ShieldGemma: B1×128 / B2×64 / B3×32 / B4×16.
Это не расширение стандартной decoder-сетки. Open не дошёл до inference из-за
недоступности оригинального checkpoint; Shield ещё не запускалась из-за отсутствия
свободной 5090. Эти попытки не дают вывода ни о достаточности памяти, ни об OOM.
У Open веса FP16, у Shield FP32; точность ради вместимости не менялась.
Предложенное зеркало Open пока не подключено к загрузчику: текущий Open-пресет
по-прежнему требует доступ к закреплённому оригиналу или его существующему кешу.

Установленная среда и кеш сохранены на сетевом диске `dng6q8yq2c` (80 GB,
EUR-NO-1). После подключения диска в `/workspace` корень комплекта:
`/workspace/u1-encoder-smoke-2026-11-09`; исходники в `repo`, среда в
`u1-env/venv`, кеш в `hf-cache`. Использовать этот абсолютный путь без переноса
venv; `HF_HOME` направить на указанный кеш. Исходный образ и версии записаны
в validation выше. После decoder-smoke сохранён Pod `2ra2t50pd79ir1`
(`u1-halo-batch-smoke-5090`), остановлен через **Stop**, не удалён.
HaloGuard-комплект и результаты: `/workspace/u1-halo-batch-smoke-2026-11-09`;
он использует тот же venv и HF-кеш из encoder-каталога.
Перед новым запуском проверка среды обязательна; GPU не оставлена работать.
Для дальнейших тестов использовать Start/Stop, не Terminate/Delete без отдельного
согласования. Остановка не резервирует карту для следующего запуска.
Сетевой диск продолжает тарифицироваться, ориентировочно $5.60/месяц.

В матрице результат каждой ячейки независим: OOM/unsupported сохраняются без
автоматического уменьшения batch size. Короткая диагностика выполняется один раз
перед матрицей, а не перед каждой стороной A/B; обязательные прогревы и подготовка
compile/ORT остаются вне таймера. Не добавлять отдельный полный диагностический
прогон ради single-to-batch сравнения: оно строится из сохранённых ответов.

`report` и `table` работают прежними командами, но показывают бакет и batch size.
Основные числа относятся к целому батчу; throughput не подменяет latency.
Отметка «ускорение во всей сетке» относится только к объявленной сетке и пройденным
шумовым проверкам; порог ускорения, flips и quality оцениваются отдельно.
Разметка по-прежнему оценивается только при batch 1, даже если speed-конфиг батчевый.

## 4. Архив, отчёт и общая таблица

```bash
./u1-rtx-env/venv/bin/python tools/benchmark.py report --output runs/gliner-baseline-1
./u1-rtx-env/venv/bin/python tools/benchmark.py report --output runs/gliner-compile-1
./u1-rtx-env/venv/bin/python tools/benchmark.py table --inputs runs/gliner-baseline-1.zip runs/gliner-compile-1.zip --output baseline-table.json
```

`report` пересобирает отчёт из каталога без GPU; `table` читает каталоги/ZIP
и сохраняет общую таблицу. `summary.json` — статусы/метрики, `REPORT.md` —
короткая сводка. Для сравнения важны все обязательные бакеты, шум и согласованность,
а не один удачный процент. Статусы:
- `smoke_passed` / `preflight_passed` — проверка запуска/контракта.
- `pilot_complete` — короткий путь завершён, pass ускорения не оценивался.
- `baseline_complete` — полный baseline при стабильных вердиктах, не гарантия отсутствия latency-шума; `baseline_unstable` отмечает нестабильные вердикты.
- `screening_pass` — конкретное сравнение прошло действующие критерии, не независимое подтверждение или safety-approval.
- `quantization_measured` — измерено квантование, качество не принято автоматически.
- `inconclusive`, `incomplete`, `rejected`, `rejected_candidate_execution` — не pass.

Архив называется по полному имени каталога: `runs/gliner.fp32` → `runs/gliner.fp32.zip`.
Конфликт каталога/ZIP останавливает запуск, существующий архив не перезаписывается.
Лимит 10 часов; таймаут/ошибка оставляют завершённые стадии и частичный архив.
Ошибка упаковки не скрывает исходную ошибку: проверить `archive_status.json` и каталог.
CLI выводит безопасный короткий JSON с путями и типом ошибки, без произвольного текста
исключений; подробная чувствительная диагностика остаётся в `.private`.

Сохранить архивы, включая неудачные попытки; после выгрузки остановить простаивающую GPU.
`table` не включает smoke и разделяет RTX 4090/5090/T4, lock, версии корпуса, scoring
и runtime-настройки. UUID отдельной VM не создаёт новую группу оборудования.
Нельзя ранжировать safety-качество моделей по speed-таблице или молча смешивать группы.
Новый паспорт precision также участвует в разделении групп. `precision_status`:
`recorded_consistent` — паспорт согласован между workers, `historical_unrecorded` —
в старом Laya-архиве он отсутствовал. Последний статус не дополняется догадкой по GPU.
В параметрическом/квантованном отчёте `precision_changed` показывает различие сторон;
это не quality-approval и не отмена существующих критериев.

## 5. Внешний candidate и квантованный артефакт

Скопировать baseline-пресет своей модели, сохранить `schema_version: 3` и его
`data`, добавить `candidate`. [Пример адаптера](../examples/u1_external_adapter.py) —
**CPU-fixture**, не реальный runtime: заимствовать интерфейс, не фиктивные вычисления.

```json
{
  "candidate": {
    "module": "/absolute/path/my_adapter.py",
    "source_sha256": "SHA256_ADAPTER",
    "dependencies": {"torch": "2.11.0+cu130"},
    "options": {}
  },
  "comparison": "engineering"
}
```

Это фрагмент, добавляемый в копию полного пресета. Выбрать `engineering`,
`parameter` или `quantization` по смыслу изменения, не ради более лёгкого порога.
`dependencies` перечисляет фактически используемые пакеты с точными версиями.
Runner сверяет их, но не устанавливает; изменение среды требует отдельного решения.

Модуль предоставляет `create(*, model, spec, directory, contract)`; возвращённый адаптер:
`contract`, `limit`, `metadata`, `compile_count` и методы `index`, `request`,
`verify`, `prepare_backend`, `boundary`, `reject_over_limit`, `size`, `close`.
Для формата 3 `index/request/verify/prepare_backend` принимают исходные сообщения,
не заранее отрендеренную строку. Для скоростных GPU-запусков также обязателен
`prove_backend`: worker вызывает его на одном полном входе вне latency и требует
подтверждения фактического backend. Автоматического fallback на профайлер worker
не предоставляет; без метода preflight завершится ошибкой.
Исключение `test_fixture` предназначено только для CPU-тестов интерфейса и не даёт
GPU-approval.
Подготовка backend вне таймера не должна кешировать входы/ответы вместо обработки.
`close` снимает патчи и восстанавливает исходную реализацию.

Для batch >1 внешний адаптер дополнительно предоставляет:
- `request_batch(payloads)` возвращает список нативных ответов в порядке входов;
- `verify_batch(payloads, indices)` проверяет фактические входы forward и возвращает ответы;
- `prepare_batch_backend(batches)` готовит используемые формы вне таймера;
- `prove_batch_backend(payloads)` возвращает доказательство с непустым списком CUDA
  `kernels` (либо сохраняет его в `metadata.batch_backend_proof`);
- `reject_batch_over_limit(payloads)` доказывает отклонение превышения до forward.

Обязателен `metadata.batch_policy`: `padding: "longest"`, `padding_side: "left"`
либо `"right"`, `pad_token_id` фактического tokenizer, `truncation: false`,
`sort_by_length: false`. Batch-путь не должен быть циклом одиночных inference;
подготовка сообщений по отдельности перед общим forward допустима. Legacy-адаптер
без этих методов поддерживает только batch 1. Паспорта precision остаются обязательными
по прежним правилам. Хеши исходников и зависимости внешнего адаптера фиксируются
как раньше; изменение файлов после начала запуска останавливает соответствующую работу.

Для нового Laya-адаптера обязателен фактический паспорт `metadata.precision`:

```json
{
  "parameter_dtypes": ["float32"],
  "autocast_enabled": true,
  "autocast_dtype": "bfloat16",
  "policy": "explicit"
}
```

Значения брать из загруженного runtime, не копировать пример как ожидаемый режим.
`parameter_dtypes` — непустой отсортированный список уникальных dtype параметров;
при выключенном autocast его dtype равен `null`. `policy` — `native_sdk` при выборе
SDK либо `explicit` при явной настройке адаптера. Название политики не заменяет
проверку dtype; инженерная пара должна совпадать по вычислительным полям.
Изменение паспорта после загрузки останавливает worker, в том числе quality;
отсутствие паспорта не заменяется dtype из конфига или названием GPU.

Ответ: `label` с native verdict; при наличии `scores` — именованные числа и
`score_type`: probabilities / independent_probabilities / logits / raw.
Допустимы `category`, `generated_tokens` и предусмотренные диагностические scores.
Не выдумывать отсутствующие вероятности и не сводить классы Koala к бинарным.
Исходник фиксируется хешем и попадает в архив; это доверенный Python-код, не sandbox.

Для квантования добавить в `candidate` поле `artifact`:

```json
{
  "path": "/absolute/path/quantized_artifact",
  "sha256": "DIGEST_FILE_INVENTORY",
  "base_checkpoint": "CHECKPOINT_FROM_MODEL_CONTRACT",
  "base_revision": "REVISION_FROM_MODEL_CONTRACT",
  "method": "METHOD_USED_BY_U2",
  "bits": 8,
  "runtime": "EXACT_RUNTIME_AND_VERSION"
}
```

Число bits и остальные значения заменить фактическими. Хеш — SHA256 канонического
словаря имён файлов и их SHA256, не произвольного ZIP. Получить значения без GPU:

```bash
./u1-rtx-env/venv/bin/python -c 'from u1_benchmark.core import sha; print(sha("/absolute/path/my_adapter.py"))'
./u1-rtx-env/venv/bin/python -c 'from u1_benchmark.contracts import file_inventory; from u1_benchmark.core import digest; print(digest(file_inventory("/absolute/path/quantized_artifact")))'
./u1-rtx-env/venv/bin/python -c 'from u1_benchmark.contracts import model_contract; print(model_contract("gliner", 3))'
```

База checkpoint обязана совпадать с контрактом; tokenizer, роли, prompt/schema,
задача и tie/stop rules не меняются скрыто. Артефакт весов не архивируется.
Использовать те же команды preflight → pilot → run, заменив путь конфигурации
и каталоги результатов. Интерфейс проверен CPU-fixture; реальная GPU-совместимость
квантованного артефакта проверяется после получения от У2.

## 6. Другой корпус скорости

Импорт локального JSONL с теми же сообщениями для всех участвующих моделей.
Пример описания `import-speed.json`:

```json
{
  "schema_version": 3,
  "source": "/absolute/path/pool.jsonl",
  "kind": "speed",
  "version": "team-speed-1",
  "fields": {"id": "id", "messages": "messages", "provenance": "provenance"},
  "models": ["halo", "gliner", "koala", "llama", "laya", "hive", "open", "shield"],
  "provenance": "Источник и разрешение на использование",
  "include_texts": false
}
```

```bash
./u1-rtx-env/venv/bin/python tools/benchmark.py prepare-data --spec import-speed.json --output data/new-speed
```

Для старого текстового JSONL заменить mapping `messages` на `text`, не задавая оба.
Индексация CPU, без весов; требуются доступ к закреплённым tokenizer и SDK из lock.
Результат: общий corpus/manifest, модельные индексы и списки ID бакетов.
Нехватка 400 эффективных входов на модельный бакет → `not_ready`, не ослабление границ.
Баланс/диапазон длины отражаются в отчёте, фиксированные Wikipedia-квоты не навязываются.

В копии пресета заменить `data` на путь нового набора и SHA256 его manifest.
Одинаковые данные и порядок сохраняются внутри пары; иной состав моделей/сообщений
или контракт означает новую версию, а не переиспользование старого индекса.
Повторная сборка исторического корпуса из локальных исходных pool/shards не нужна
обычному пользователю: команда получает готовый закреплённый выпуск через `fetch-data`.

## 7. Отдельный quality-набор

Набор готовит У2; сейчас настоящего набора нет. Он не блокирует передачу измерителя.
Для импорта строка содержит id, messages (или text), target, target_role user/assistant,
при необходимости target_index и словарь slices с одинаковыми именами во всех строках.
Пример `import-quality.json`:

```json
{
  "schema_version": 3,
  "source": "/absolute/path/quality.jsonl",
  "kind": "quality",
  "version": "team-quality-1",
  "fields": {
    "id": "id", "messages": "messages", "target": "target",
    "target_role": "target_role", "slices": "slices"
  },
  "provenance": "Источник, разметка и разрешение на использование",
  "include_texts": false
}
```

```bash
./u1-rtx-env/venv/bin/python tools/benchmark.py prepare-data --spec import-quality.json --output data/team-quality
```

В копии своей конфигурации добавить `quality_data` с path/manifest_sha256 и
`label_mapping`. Ниже **фрагмент для явно согласованной бинарной задачи** GLiNER;
не универсальный mapping Koala или другой политики:

```json
{
  "quality_data": {"path": "data/team-quality", "manifest_sha256": "SHA256_QUALITY_MANIFEST"},
  "label_mapping": {
    "targets": {"benign": "negative", "harmful": "positive"},
    "verdicts": {"safe": "negative", "unsafe": "positive"},
    "labels": ["negative", "positive"],
    "positive_label": "positive"
  }
}
```

Названия targets заменить метками реального набора по согласованному смыслу.
Нельзя оценивать метку ответа предсказанием по запросу HaloGuard: `target_mismatch`,
`not_applicable`, over-limit и ошибки видны в coverage, а не исчезают из набора.
Для quality нет скоростных квот и отбора удобных длин.

```bash
./u1-rtx-env/venv/bin/python tools/benchmark.py quality --config my-quality.json --environment ./u1-rtx-env --output runs/quality-1
```

`quality` работает отдельно от latency и не обращается к неиспользуемому speed-корпусу.
Конфигурация и модельный артефакт фиксируются так же, как для скорости.
Считаются confusion matrix, accuracy, macro-F1, N/coverage по срезам;
FPR/FNR только для явной бинарной задачи, нулевой знаменатель → `null`.
Качество без согласованного правила не объявляется сохранённым.
До настоящего набора `quality_evaluated=false`; CPU-fixture проверяет лишь интерфейс.

По умолчанию тексты, сообщения, отрендеренные промпты и чувствительные логи не входят
в архив. `include_texts: true` допустим только для разрешённых к распространению данных.
ID, targets и имена срезов тоже не должны содержать персональные данные.

## 8. Ответственность и история

- Владельцы моделей: RTX setup, доступ HF, проверка своей модели и полные baseline/comparison-архивы.
- У1: инструмент, speed-корпус, протокол, помощь с воспроизведением blocker и общая таблица.
- У2: квантованный артефакт, размеченный набор и согласование quality-требований с командой/куратором.
- У3/У4: автоподбор и CLI; интеграция отдельным этапом. Новые наши бенчмарки не условие передачи.

Исторические инструкции и approvals не являются разрешением нового запуска:
[приёмка v2/T4](u1_handoff_readiness.md#историческая-приёмка-v2--t4),
[инструкция до актуализации](https://github.com/kekwak/HiveTrace-AutoTune-Guardrails/blob/23a4874a183f0815e22e4b33b785d13c16e5c7be/docs/u1_handoff.md),
[screening 30–31 октября](benchmark.md), [разбор профилирования](../README.md#профилирование-почему-выбрали-эти-оптимизации).

Текущий необязательный Kaggle launcher собирается через `tools/package_current_smoke.py`;
для Open обязателен проверенный `--dataset-overlay data/speed/role-corpus-open-task-v2`
до создания пакета. Диагностика Open native/task использует отдельный
`tools/diagnose_open_precision.py --input-profile native|task` и не получает approval.
Значение с `|` — выбор одного профиля, не shell-команда.

Launcher не содержит новой измерительной логики. Полный встроенный исходник превышает
лимит Quick Save Kaggle 1 МБ; это известное ограничение сохранения notebook, не blocker
консольного запуска из клона. Проверенный архив 4 ноября сохранён; GPU остановлена.
