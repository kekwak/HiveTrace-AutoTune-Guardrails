# У1: общий измеритель и передача участникам

## Актуальный комплект: роли и RTX, 1 ноября 2026

**Точка входа из GitHub:** исходники, 11 конфигураций `configs/benchmark-v3` и четыре
lock-файла находятся в репозитории; ZIP не нужен. Новый [HF-выпуск v4](https://huggingface.co/datasets/vXofi/guardrails-speed-en-ru/tree/v4)
закреплён в `configs/data/role_speed_v4.json` на commit
`7324b8823ded971c05076e0c7f4ace5d9b867863`. Исторический тег v3 не перемещён.

Новый формат конфигурации **`schema_version: 3`** использует общий корпус сообщений и
независимые модельные бакеты. Старый формат 2, опубликованный корпус v3, T4 lock и
исторические измерения сохранены. Старые approvals не распространяются на новые входы.
GPU-приёмку RTX выполняют владельцы моделей; наша текущая приёмка — CPU и индексация.

### Установка и доступ

Linux x86_64 / Python 3.13; GPU 0 должна быть RTX 4090 или RTX 5090. Из корня клона:

```bash
python tools/benchmark.py setup --profile team-rtx --environment ./u1-rtx-env
./u1-rtx-env/venv/bin/python tools/benchmark.py auth-check --model shield
```

97 точных wheels закреплены в `env/team_rtx.lock.json`/`.txt`: прежние 96 артефактов без
изменений плюс `laya==0.3.20`, без extras. Проверки объявленных зависимостей выполнены
на CPU; повторная установка RTX lock, Laya inference и GPU-совместимость **не проверены**.
CUDA 13 PyTorch и CUDA 12 библиотеки ORT сосуществуют как в исходном lock. Не менять
версии вручную ради успешного запуска; сохранить конкретный blocker. Старый default
`setup` по-прежнему выбирает T4. Предупреждение host `sitecustomize/wrapt` не исправляется
добавлением случайных зависимостей.

На машине владельца используйте его HF login/cache либо `HF_TOKEN`/`HF_TOKEN_PATH`,
заданные до запуска процесса. Токен не передаётся в аргументах, JSON или архиве и не
копируется автоматически с этого компьютера. `auth-check` читает только метаданные
конфигурации и одного файла весов; весов и GPU-проверки нет. Локальная проверка Shield
1 ноября успешна, это не авторизация чужого сервера.

### Нативные входы

Модель | Применимые сообщения и цель
--- | ---
GLiNER / Koala / Llama / Laya | Если есть assistant, только его текст; иначе user. Нативная подготовка/SDK.
HaloGuard | Последний user через авторский prompt, thinking off и `Safety: `. В паре assistant игнорируется; assistant-only неприменим.
HiveTraceGuard-Pro | Собственный шаблон: цель — последнее сообщение, предшествующий user сохраняется как контекст. Assistant-only технически принимается шаблоном, качество без контекста не подтверждается.
ShieldGemma | User-only или user+assistant; во втором случае оценивается ответ. Авторские Dangerous Content guidelines различаются для prompt/response. Assistant-only неприменим.
OpenGuardrails 0124 | Только user-only, собственный шаблон закреплённой ревизии, `add_generation_prompt=True`, greedy generation до 10 токенов. Режимы ответа неподтверждены, в основной замер не включаются.

OpenGuardrails 0124 не использует шаблон 2510 в новом пути. Исторические результаты с
2510 не переписываются и не являются baseline для новой подготовки. Источники:
[0124](https://huggingface.co/openguardrails/OpenGuardrails-Text-4B-0124#quick-start),
[HaloGuard](https://huggingface.co/astroware/HaloGuard1-Gen-0.8B#highlights),
[HiveTrace](https://huggingface.co/hivetrace/HiveTraceGuard-Pro#guard-modes),
[ShieldGemma](https://huggingface.co/google/shieldgemma-2b#usage-with-chat-templates).

### Корпус и модельные бакеты

Запись содержит `id`, `messages: [{role, content}]`, `provenance`, `content_sha256`;
допустимы user-only, assistant-only и user+assistant. Старое поле `text` при импорте
означает один user. Новые роли/содержимое получают новый ID; исходный ID остаётся в
происхождении. Синтетическое назначение роли не выдаётся за настоящий ответ LLM.

Бакеты полных входов: B1 1–512, B2 513–1024, B3 1025–2048, B4 2049–4096. По 400
уникальных эффективных входов **на модельный бакет**, не по 400 на каждый формат.
Энкодеры — B1; декодеры — B1–B4. Один ID может иметь разные бакеты разных моделей.
Нет пересечений между моделями и искусственного padding/усечки. Различия игнорируемого
контекста не увеличивают количество уникальных входов. Восемь равных интервалов длины
имеют цель 50 записей; дефициты перераспределяются, недостижимые длины видны в отчёте.
Язык/применимый формат балансируются по доступности, seed 42; фиксированных Wikipedia-квот нет.

Сборка из сохранённых исходников, только CPU, без весов; нужны точные tokenizer/SDK
версии из контрактов, включая Laya. Отдельная локальная среда подготовки не заменяет RTX lock:

```bash
python tools/build_role_corpus.py \
  --base data/speed/v3/releases/4bca32272b74f30b/corpus.jsonl \
  --pool data/speed/v3/pool.jsonl --output data/speed/NEW_ROLE_BUILD
```

Команда пробует 12000/24000/36000/48000 общих кандидатов, сохраняет попытки и `latest.json`.
Нехватка примеров или доступа означает `not_ready`, а не ослабление правил. Исходные
локальные pool/shard-материалы не включаются в измерительные архивы автоматически.
Для собственного JSONL используйте `prepare-data` с `schema_version: 3`, явным
`fields: {"id":"id", "messages":"messages", "provenance":"provenance"}`,
`kind: speed`, `models`, `version`, `provenance`, `include_texts` и `source`.

### Запуск владельцем

После клонирования используйте конфигурации прямо из `configs/benchmark-v3`, не из
ZIP. Сначала установите среду как выше и получите закреплённый корпус:

```bash
./u1-rtx-env/venv/bin/python tools/benchmark.py fetch-data
```

Команда скачивает этот commit в `data/speed/role-corpus` и проверяет все хеши и
модельные бакеты. Можно передать `--source /path/to/verified/dataset` для локальной
копии либо `--revision` с точным commit, но manifest обязан совпасть с закреплённым
SHA256. `main`, исторический v3 и непроверенный набор не подставляются. Уже существующий
каталог не перезаписывается. Это загрузка данных, не GPU-запуск.

Конфигурация 3 использует `data: {"path":"/absolute/path/dataset",
"manifest_sha256":"SHA256 manifest.json"}` вместо исторического `profile: published`.
Остальные baseline/candidate/comparison/mode поля сохраняются. Для внешнего адаптера
`index/request/verify/prepare_backend` получают исходный список сообщений; точный
контракт передаётся в `create`. Пример интерфейса поддерживает обе версии, но остаётся
CPU fixture, не реализацией квантования.

```bash
python tools/benchmark.py preflight --config configs/benchmark-v3/gliner_compile.json --environment ./u1-rtx-env --output runs/preflight-1
python tools/benchmark.py run --pilot --config configs/benchmark-v3/gliner_compile.json --environment ./u1-rtx-env --output runs/pilot-1
python tools/benchmark.py run --config configs/benchmark-v3/gliner_compile.json --environment ./u1-rtx-env --output runs/screening-1
python tools/benchmark.py report --output runs/screening-1
python tools/benchmark.py table --inputs runs/screening-1.zip --output baseline-table.json
```

Все сравнения внутри одной модели получают одинаковые ID, сообщения, token IDs/маски и
порядок. Токенизация остаётся внутри полного таймера; индекс не заменяет обработку
сообщений. Сохраняются A/A, AB/BA, свежие worker-процессы, 10-часовой лимит, memory/profile.
Baseline-only теперь показывает p95 каждой серии, дрейф max/min−1, A/A envelope и
блочные интервалы p50/p95 (2000 повторов, seed 42). Это диагностика по трём сериям,
не новый порог стабильности. `baseline_complete` — комплектность, не отсутствие шума.
GPU, драйвер, lock, корпус и режимы разделяют группы совместимости; скорости разных
моделей на разных модельных выборках нельзя считать сравнением идентичных нагрузок.

### Набор качества от другого участника

Он пока не готов и не блокирует передачу инструмента. Импорт 3 принимает сообщения
или text, `target`, обязательный `target_role` и необязательный согласованный `target_index`,
а также срезы. В паре размечается конкретный запрос или ответ. Если модель оценивает
другое сообщение, строка получает `target_mismatch`; неподдерживаемый режим —
`not_applicable`. Обе причины видны в coverage, как over-limit/errors. Никаких
скоростных квот или отбора удобных длин. Label mapping остаётся явным.

Quality запускается отдельно прежней командой `quality`, вне latency. Пока реального
набора нет, примеры проверяют только интерфейс; реальные запуски имеют
`quality_evaluated=false`. Закрытые сообщения и отрендеренные промпты по умолчанию
не архивируются. Статусы данных, CPU-подготовки и GPU-совместимости не объединяются.

## Исторический комплект v2 / T4

Статус на 31 октября: реализация v2 готова, 208 CPU-тестов завершились успешно (один пропущен). Новый GPU-smoke прошли HaloGuard, GLiNER, Koala, Llama Prompt Guard, HiveTraceGuard-Pro и OpenGuardrails. Laya блокируется отсутствующим SDK в lock, ShieldGemma — gated-доступом; подробности в [матрице готовности](u1_handoff_readiness.md). Полные новые baseline остальных моделей снимают владельцы. Реальный квантованный артефакт не проверен: проверен интерфейс с явно тестовым CPU-адаптером. Это не создание safety-набора или квантование.

## Быстрый старт

Linux x86_64, Python 3.13, T4, интернет для первого скачивания. Команды из корня клона. Используется только GPU 0. `setup` устанавливает неизменённый lock в отдельный venv; host Python/PyTorch не обновляются.

```bash
python tools/benchmark.py setup --environment /kaggle/temp/u1-env
python tools/benchmark.py preflight --config configs/benchmark-v2/koala_baseline.json --environment /kaggle/temp/u1-env --output runs/koala-preflight
python tools/benchmark.py run --config configs/benchmark-v2/koala_baseline.json --environment /kaggle/temp/u1-env --output runs/koala-baseline
python tools/benchmark.py report --output runs/koala-baseline
python tools/benchmark.py table --inputs runs/koala-baseline.zip runs/other-baseline.zip --output baseline-table.json
```

Для первого короткого smoke в копии конфигурации поставить `"mode": "smoke"`. Этот режим ограничен часом на модель: два реальных текста из каждого обязательного бакета, проверка полного запроса/входов, A→B→A, точный лимит и отказ до forward на превышении, backend proof, latency и память процесса. Это не baseline и не screening. `preflight` с обычной конфигурацией проверяет весь применимый корпус, но не запускает latency-серии.

`run --pilot` оставляет полный preflight, сокращает A/B до 12 запросов и одной пары; pass не выдаётся. Новый каталог нужен для каждой попытки. После запуска скачайте соседний ZIP, включая ошибки; `summary.json` содержит статусы, `REPORT.md` короткую таблицу. Не оставляйте GPU включённой после скачивания.

Kaggle допускает тонкий launcher, но все вычисления остаются в консольном скрипте. `tools/package_u1_handoff.py --output benchmark_runs/NEW_NAME.zip` механически собирает источники, checksums и приватный launcher для восьми smoke. Это не новый измерительный notebook. Старые presets `halo_fla`, `gliner_compile`, `gliner_ort` и их результаты используют прежний формат v1.

## Конфигурация v2

```json
{
  "schema_version": 2,
  "model": "gliner",
  "baseline": {"backend": "eager"},
  "candidate": {"backend": "compile"},
  "comparison": "engineering",
  "data": {"profile": "published"},
  "mode": "screening",
  "seed": 17,
  "timeout_seconds": 36000
}
```

Без `candidate` выполняются baseline-only preflight, A/A 2×30, три серии полного бакета, три memory-прогона и short/long-профили. Фиктивной второй модели и статуса ускорения нет. Для Halo доступен кандидат `fla`, для GLiNER `compile`/`ort`; исходные модельные реализации переиспользованы. Другие кандидаты подключаются явно через локальный модуль.

Сравнение: 20 прогревов перед каждой стороной, A/A 2×30, A/B 3×400 на сторону в каждом бакете, AB/BA, одинаковые ID/порядок в паре; новая модель и процесс для каждой стороны. Полный путь text → CPU scores/verdict; ни токенизированные входы, ни ответы не кешируются вместо обработки текста. Подготовка всех используемых worker форм и backend proof вне latency. ORT оставляет голову GLiNER и `metadata-cpu-v1`; провал проверки размещения не маскируется CUDA EP.

`engineering`: p95 ниже строго более чем на 20% во всех бакетах, ноль flips и нестабильности, проверки шума. `parameter`: то же с 10%. `quantization`: отдельно эффект скорости, изменения уникальных ID, направления и макросреднее долей по бакетам; общего принятия без quality-допусков нет. Никакие результаты не являются независимым подтверждением или доказательством безопасности.

## Подключить свою модель/артефакт

Точка расширения: `create(model, spec, directory, contract)` в явно указанном `.py`. [Исполняемый пример интерфейса](../examples/u1_external_adapter.py) намеренно тестовый: не загружает веса, не является квантованием, помечает результат `fixture_complete`, а не GPU-pass. Переносите интерфейс, а не фиктивные вычисления.

Адаптер предоставляет `contract`, `limit`, `metadata`, `compile_count`, методы:

- `index(text)`: полные длины, SHA256 token IDs/масок и их пары, user_tokens; без усечки и GPU-forward.
- `request(text)`: каждый раз подготовить текст, выполнить реальный runtime, вернуть CPU-ответ.
- `verify(text, expected)`: проверить не только отдельную токенизацию, но и вход фактического запроса.
- `prepare_backend(texts)`: экспорт/compile и подготовка используемых форм вне измерения; без изменения задачи.
- `prove_backend(text)`: сохранить доказательство реального backend и ограничений размещения; профайлер не оставлять активным.
- `boundary()`, `reject_over_limit(text)`: точный лимит и отказ до вычислений при превышении.
- `size()`: веса, уникальные хранилища параметров/буферов и экспорт, без смешивания с VRAM; неизвестное обозначать `null`.
- `close()`: снять патчи/хуки, восстановить baseline-реализации, освободить runtime.

`request` возвращает `label` (исходная метка), необязательно `scores` (словарь имён и чисел), `score_type` (`probabilities`, `independent_probabilities`, `logits`, `raw`), `category`, `generated_tokens`. Нет вероятностей у generative API → не добавлять вымышленные. Koala остаётся девятиклассовой, Laya A/B, Shield Yes/No. Native labels не приводятся автоматически к бинарной задаче.

В `baseline`/`candidate` указать `module` (локальный путь), `source_sha256`, `dependencies` (точные версии пакетов, допускается `{}`), при необходимости `options`. Это доверенный Python-код, не sandbox: не подключать непроверенные модули и не записывать токены в исходник или конфигурацию. Модуль фиксируется в архиве; зависимости сверяются, а не устанавливаются автоматически. Дополнительные локальные helpers следует включить в этот один модуль либо оформить версионированным установленным пакетом.

Для `comparison: quantization` кандидат дополнительно содержит `artifact`: `path`, `sha256`, `base_checkpoint`, `base_revision`, `method`, `bits`, `runtime`. `sha256` здесь SHA256 канонического JSON-словаря относительных имён файлов и их SHA256 (`digest(file_inventory(path))`), а не хеш произвольного ZIP. Артефакт не архивируется. Его база обязана совпадать с закреплённым checkpoint; tokenizer, prompt/schema, задача и tie/stop rules сохраняются. Runtime, precision и веса меняются явно. Реальный smoke такого артефакта остаётся обязанностью проверки после его получения.

Получить значения для конфигурации можно без GPU:

```bash
python -c 'from u1_benchmark.core import sha; print(sha("/absolute/path/my_adapter.py"))'
python -c 'from u1_benchmark.contracts import file_inventory; from u1_benchmark.core import digest; print(digest(file_inventory("/absolute/path/quantized_artifact")))'
python -c 'from u1_benchmark.contracts import model_contract; print(model_contract("gliner"))'
```

В копии конфигурации baseline добавить candidate с этими значениями и `comparison: quantization`, затем:

```bash
python tools/benchmark.py preflight --config my-quantized.json --environment /kaggle/temp/u1-env --output runs/my-quantized-preflight
python tools/benchmark.py run --pilot --config my-quantized.json --environment /kaggle/temp/u1-env --output runs/my-quantized-pilot
python tools/benchmark.py run --config my-quantized.json --environment /kaggle/temp/u1-env --output runs/my-quantized-screening
```

Не запускать последний шаг при неуспешном preflight/pilot. Новые зависимости требуют отдельного решения по среде: runner не делает `pip install` по декларации адаптера.

## Другой скоростной корпус

Создать описание импорта:

```json
{
  "source": "/absolute/path/pool.jsonl",
  "kind": "speed",
  "version": "team-speed-1",
  "fields": {"id": "id", "text": "text"},
  "models": ["halo", "gliner", "koala", "llama", "laya", "hive", "open", "shield"],
  "provenance": "Источник и разрешение на использование",
  "include_texts": false
}
```

```bash
/kaggle/temp/u1-env/venv/bin/python tools/benchmark.py prepare-data --spec import.json --output data/new-speed
```

Команда не загружает веса: проверяет tokenizer/preparation и строит пересечения полных входов, seed 42. Нужны tokenizer-доступ и SDK всех перечисленных моделей; нехватка пакета или доступа блокирует индексацию без установки новой версии. Для набора всех восьми моделей Laya требует отдельного решения по SDK в lock.

Результат: `manifest.json`, общий `corpus.jsonl`, индексы моделей, общие списки ID. Для screening ровно 400 уникальных текстов в каждом обязательном бакете; нехватка → `not_ready`. В smoke/pilot меньший непустой набор допустим, но не получает pass ускорения. Квоты Wikipedia/языков не навязываются новым данным. Неподходящие кандидаты сохраняются в исходном пуле и индексах; бакет выбирается по пересечению, без padding/усечки.

В конфигурации заменить `data` на `{"path": "/absolute/path/data/new-speed", "manifest_sha256": "SHA256 manifest.json"}`. Поменяли тексты, контракты или код подготовки → новый manifest и версия; старый индекс использовать нельзя. Формальный общий набор команды содержит всех участников; набор с меньшим списком моделей имеет другую идентичность и не объединяется с ним.

## Отдельные размеченные данные

Импорт `kind: quality` требует полей `id`, `text`, `target`; `slices` задаёт столбец со словарём именованных срезов. Во всех строках должны присутствовать одни и те же имена срезов; неизвестное значение отмечается явно. Нет квот по длине/языку и отбора по удобным длинам.

Подключить `quality_data` с path/manifest_sha256 и `label_mapping`:

```json
{
  "targets": {"dataset_negative": "negative", "dataset_positive": "positive"},
  "verdicts": {"safe": "negative", "unsafe": "positive"},
  "labels": ["negative", "positive"],
  "positive_label": "positive"
}
```

Это шаблон явного согласования задачи, не утверждение, что любые метки можно свести к safe/unsafe. Для Koala нужно описать исходные классы; для другой safety-политики такой mapping может быть неверен. Без mapping или при неизвестной метке оценка недоступна, ничего автоматически не бинаризуется.

```bash
python tools/benchmark.py quality --config my-comparison.json --environment /kaggle/temp/u1-env --output runs/quality-1
```

Quality можно выполнить без speed-этапов. В архиве остаются точные конфигурации/хеши артефактов, ID, text SHA256, результаты, confusion matrix, accuracy/macro-F1 и N/coverage по всем срезам. Для явной бинарной задачи дополнительно FPR/FNR; нулевой знаменатель → `null`, не ноль. Ошибки и over-limit записываются на каждую строку, а не исчезают из знаменателя покрытия. Неразмеченный speed-корпус не используется для этих метрик.

По умолчанию исходные тексты не копируются в архив; stdout/stderr закрытых запросов подавляются, подробная потенциально чувствительная диагностика остаётся в `.private` и не архивируется. `include_texts: true` следует включать только для данных, которые разрешено распространять. ID, targets и названия срезов тоже выбирайте без персональных данных. Адаптеры доверенные: это защита от случайных логов, а не от вредоносного кода.

## Статусы и передача

`smoke_passed` не включается в baseline-таблицу. `baseline_complete` означает выполненный baseline, не ускорение; `screening_pass` относится только к конкретному сравнению и корпусу. `quantization_measured` не означает принятие качества. `inconclusive`/`incomplete`/`rejected_candidate_execution` не становятся pass. Архивы ошибок сохраняются.

Общая таблица группируется по lock, manifest корпуса и фактическому паспорту среды (GPU, драйвер, версии, настройки потоков/TF32), содержит исходные scoring/dtype, p95/p50, память процесса и отдельно Torch counters/размер. UUID конкретной VM в группу совместимости не входит. Не сравнивать разные группы молча. Память процесса измеряется опросом примерно каждые 50 мс: короткий пик может быть пропущен; фактические интервалы и недоступные показатели записаны.

Перед передачей проверить [матрицу готовности](u1_handoff_readiness.md). Полные baseline шести остальных моделей, размеченные данные/допуски У2 и реальный квантованный артефакт остаются отдельными действиями. Публикация и CLI/autotune не включены.
