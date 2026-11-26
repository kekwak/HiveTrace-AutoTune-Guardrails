# У1: запуск измерителя v4

## Общий CLI: plan → eval → report

Точка входа команды — `guardrail-tuner` из клона. CLI-план версии 2 содержит
полный конфиг измерителя schema 4; протокол измерения не меняется. Модели работают
в отдельных закреплённых venv, не в среде самого CLI.

```bash
uv sync --locked --python 3.12 --extra hub
```

Установите `team-rtx` для ProtectAI и `team-d3` для AccuKnox по разделам ниже,
восстановите индексы. В полном конфиге укажите `environment` и `data.path` своей
машины; checkpoint/scoring/manifest ради демонстрации не меняйте. Не устанавливайте
CLI в модельный venv: его inventory проверяется строго. HF-токен не записывается
в конфиг; используйте существующий кеш или переменные среды.

ProtectAI:

```bash
uv run --locked --no-sync guardrail-tuner plan --config configs/v4/protectai.json --output protectai-plan.json
uv run --locked --no-sync guardrail-tuner eval --config protectai-plan.json --output runs/protectai-eval
uv run --locked --no-sync guardrail-tuner report runs/protectai-eval
uv run --locked --no-sync guardrail-tuner run --config protectai-plan.json --text "The meeting starts at noon."
uv run --locked --no-sync guardrail-tuner serve --config protectai-plan.json
```

AccuKnox:

```bash
uv run --locked --no-sync guardrail-tuner plan --config configs/v4/accuknox.json --output accuknox-plan.json
uv run --locked --no-sync guardrail-tuner eval --config accuknox-plan.json --output runs/accuknox-eval
uv run --locked --no-sync guardrail-tuner report runs/accuknox-eval
uv run --locked --no-sync guardrail-tuner run --config accuknox-plan.json --text "The meeting starts at noon."
```

Без reference `eval` выполняет штатный baseline и один search-кандидат.
С `--baseline-ref runs/protectai-eval/baseline` или ZIP полного baseline исполняется
только candidate. Несовместимость — ошибка, не скрытый повторный baseline.
Объём серий и батчи берутся из конфига; отдельного демонстрационного pilot нет.

Правила выбираются по capabilities: для HF classifier возможны c12/c8/u4/u1;
для D3 — c12 и проверка c7. u1 применяется только к batch=1, native FLA D3
отмечается `already_active`. Применение записывается в worker-evidence, ошибки
применимого правила не включают fallback. `plan MODEL` без полного конфига
сохраняет старый metadata-only интерфейс и не заменяет измерительный план версии 2.

`run/serve` выбирают candidate; `--variant baseline` выбирает исходную модель.
`run --messages conversation.json` принимает один массив role/content сообщений.
HTTP `POST /infer`: `{"inputs":["text"]}` либо `{"messages":[...]}`.
Сервис держит одну модель, без baseline-прогонов на каждом запросе.

Результат `eval`: `report.json`, `report.md`, архивы и стадии `baseline/`,
`candidate/`, при разметке `quality/`. Исходный план не переписывается.
Локальный `.private/evaluated_plan.json` сохраняет вариант с аннотацией результата
и пригоден для `run/serve`; он исключён из публичного архива. Известные flips или
нестабильность инженерного варианта блокируют исполнение аннотированного плана.
Отсутствие speedup не блокирует исполнение. Незамеренный план остаётся unverified.

`quality_data`, `label_mapping`, `quality_batch_sizes` — существующие поля У1.
Quality выполняется отдельно от latency. Для явного mapping safe/unsafe модуль
команды агрегирует сохранённые предсказания без повторного инференса; multiclass
остаётся в отчёте У1. Без набора — `quality_evaluated=false`, автоматического
quality-approval нет.

Старый `optimize` сохраняет прежний алгоритм и `config.yaml`, дополнительно
выдаёт `execution.json` с сохранённым артефактом для `run/serve`. ONNX external
data, provider/options и checksums сохраняются; повторного экспорта нет.
Историческая подготовка, включая явно записанную усечку, не выдаётся за новый
протокол без усечки. Переносите каталог артефактов вместе с execution.json;
не публикуйте веса или приватные источники вместе с отчётом.

Статус интеграции на 26 ноября: CPU-путь проверен, CI проходит на Python 3.10/3.12
и в отдельной среде У1. ProtectAI прошёл полный GPU-eval на RTX 5090: обе ячейки
`reference_pass`, без смен вердиктов. Полный повтор AccuKnox комплектен (8/8),
но имеет `rejected`/`below_target` при неизменных вердиктах. Это отрицательный
результат скорости, не ошибка CLI. `run/serve` обеих моделей прошли GPU-проверку
на контрольном тексте. [Числа, артефакты и границы проверки](../results/u1_handoff/2026-11-26-cli-eval/README.md).
Старые baseline сами по себе не подтверждают новые правила общего CLI.


Актуальный CLI принимает только `schema_version=4`. Форматы результатов, данных и
протокола версионируются отдельно. Новые результаты: версия 5, speed-протокол
`u1-static-batch-v4-reuse-v1`, quality-протокол 2. Полный baseline v4 поддерживается
без миграции и без изменения оригинала. Конфиги/архивы до schema 4 не принимаются.
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

## Новые семейства: ProtectAI и AccuKnox

Основной путь подключения новой модели — полный конфиг семейства, не новая ветка
по имени checkpoint. `hf_sequence_classification` загружает
`AutoModelForSequenceClassification`; `d3_decision` использует закреплённый d3 runtime.
Старые восемь адаптеров и их контракты не изменены. `plan` показывает `capabilities`
без весов: `runtime_check_required` не означает, что CUDA backend уже проверен.

| Пресет | Нативный контракт | Кандидаты для проверки |
| --- | --- | --- |
| [ProtectAI](../configs/v4/protectai.json) | FP32; SAFE/INJECTION; softmax/argmax, первый индекс при tie; assistant, если он есть, иначе user; лимит 512 | [compile](../configs/v4/protectai_compile.json), [ORT CUDA](../configs/v4/protectai_ort.json) полного tensor-forward до logits |
| [AccuKnox](../configs/v4/accuknox.json) | BF16 backbone + FP32 readout; авторский d3, block при p≥0,5; лимит 4096 | compile текстового tensor-forward; reference/FLA только linear attention; ORT не поддержан |

ProtectAI закреплён на `90c9989b1a342275dd0d1a95aad283c04e075671`, AccuKnox — на
`e4e43d89a1f2c0ae2025b50ec884d8de32d2cde3`. Исходники, шаблоны, SDK-assets и правила
ответа входят в идентичность конфигурации. Несовместимые overrides отклоняются.
За основу взяты [карточка ProtectAI](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2/tree/90c9989b1a342275dd0d1a95aad283c04e075671)
и [карточка/код AccuKnox](https://huggingface.co/Accuknoxtechnologies/accuknox-guard-1.1/tree/e4e43d89a1f2c0ae2025b50ec884d8de32d2cde3).
Обе модели заявлены для английского; RU здесь — нагрузка скорости. Различные
injection/blocking-задачи и scores между моделями не сравниваются.

AccuKnox не вызывает Qwen `generate`: на каждый независимый state приходится один
BLOCK/RESPONSE-вопрос, затем один общий батч probabilities/readout. User-only —
строка с BLOCK; пара — объект user/assistant с RESPONSE; assistant-only — объект
assistant с RESPONSE, без вымышленного запроса. Сохраняются noncausal full attention,
causal linear attention, авторский SDPA-dispatch и нативный выбор linear kernels.
`readout.safetensors` учитывается в памяти и паспорте precision отдельно от backbone.
Паспорт precision v2 хранит роли тензоров; старые паспорта не подтверждают этот режим.

### Среда и CPU-индексы

ProtectAI использует `team-rtx`. Для d3 предусмотрен отдельный `team-d3`:
Linux/Python 3.13, прежний CUDA PyTorch, Transformers 5.17.0, FLA 0.5.2.
Не устанавливайте эти версии поверх существующего venv.
[Lock и способ его сборки](../env/team_d3.README.md) готовы: 61 пакет, SHA256
артефактов и исходный structured pip report. Версии автоматически не подбираются.
Установка из lock прошла на RTX 5090 25 ноября. Первый smoke выявил CUDA-сбои;
на [другой физической 5090](../results/u1_handoff/2026-11-26-new-gpu-smoke/README.md)
с теми же исходниками и средами ProtectAI B1 и AccuKnox B1–B3 прошли с batch=2.
Индексный блокер AccuKnox B4 закрыт [пересборкой в d3-env](../results/u1_handoff/2026-11-26-d3-reindex/README.md):
повторный B4×2 smoke прошёл. Затем [полный baseline](../results/u1_handoff/2026-11-26-new-families-baseline/README.md)
завершил всю указанную сетку на RTX 5090. Последующая проверка правил общего CLI
описана в [отдельном eval](../results/u1_handoff/2026-11-26-cli-eval/README.md). Установка сама по себе
не подтверждает работу модели; CUDA-сбои первого размещения не объявлены аппаратным дефектом.

```bash
python3.13 tools/benchmark.py setup --profile team-d3 --environment ./u1-d3-env
```

Команда выше выполняется на целевой GPU-машине. Для CPU-индексации
нужны tokenizer и закреплённый код подготовки, но не веса. У d3
`execution.trust_remote_code=true` разрешает исполнение только проверенных SHA256
исходников из manifest; README не исполняется.

[Рецепт исходной сборки](../configs/v4/data_new_families.json) использует прежние
12000 сообщений с теми же ID и происхождением. Снимок от 25 ноября сохраняется,
но его D3-индекс нельзя использовать с целевой 5.17.0 без пересборки.
Текущие семь конфигов используют `data/speed/new-families-d3-20261126`, manifest
`846e234bd0dff69ca1d22d460e49b9c4dfdc85e08576f10f01f46bc0dd5cbc2c`.
Сообщения и списки ID прежние; HF-публикация не изменена.

Для запуска из чистого клона восстановите проверенные снимки без повторной
токенизации. `fetch-data` скачивает закреплённый публичный корпус; если он уже
есть, пропустите первую команду. В Git находятся только индексы, manifest и
списки ID, без дублирования текстов. Восстановление не загружает модели и не
требует GPU, проверяет SHA256 и не перезаписывает существующие снимки.

```bash
python tools/benchmark.py fetch-data --output data/speed/role-corpus
python results/u1_handoff/2026-11-26-d3-reindex/restore_data.py \
  --source data/speed/role-corpus --output-root data/speed
```

После этого семь конфигов готовы к `plan` и запуску в соответствующей среде.
Восстановлены оба снимка: первоначальный нужен только для воспроизведения
истории и повторной проверки пересборки; измерения используют `new-families-d3-20261126`.
Это доставка уже проверенных индексов, не новая GPU-проверка.

Необязательная повторная CPU-пересборка D3 в другом, ещё не существующем каталоге:

```bash
./u1-d3-env/venv/bin/python results/u1_handoff/2026-11-26-d3-reindex/rebuild.py \
  --repo . --source data/speed/new-families-20261125 \
  --output data/speed/new-families-d3-20261126-check
```

Не переиспользуйте существующий output. Для этого снимка ожидается тот же
`manifest_sha256`, что указан выше; после восстановления менять конфиги не нужно.
Для нового корпуса перенесите его `path` и `manifest_sha256` во все сравниваемые
конфиги. Продолжать можно только с готовыми обязательными бакетами;
дефицит даёт `not_ready`, а не усечённые входы.
Измеренная baseline-сетка на RTX 5090: ProtectAI B1 `[1,256]`, AccuKnox B1 `[1,64]`,
B2 `[1,32]`, B3 `[1,16]`, B4 `[1,8]`. Все 10 ячеек комплектны, без OOM;
Статус кандидатов приведён в разделе общего CLI выше. Размеры редактируются в JSON.

Команда требует сохранённую исходную сборку; её SHA256 строго проверяется.
Для нового произвольного корпуса используйте `prepare-data` отдельно в среде
соответствующей модели, а не токенизацию всех семейств одной версией Transformers.
Новая сборка: все пять модельных бакетов содержат по 400
уникальных эффективных входов; две последовательные сборки дали одинаковые файлы.
У AccuKnox B1 отсутствуют длины 1–64; гистограмма B4:
`[68,68,68,68,68,40,13,7]`. Эти пробелы не заполнялись padding.
D3 переиндексирован на Linux/Python 3.13.15 / Transformers 5.17.0 / Tokenizers 0.23.3;
индекс ProtectAI сохранён. [Сверка изменений и GPU-smoke](../results/u1_handoff/2026-11-26-d3-reindex/README.md).
Реальный worker по-прежнему сверяет каждый фактический вход с индексом и останавливается при несовпадении.

### Запуск владельцем GPU

После подготовки среды/индексов используйте существующий протокол:

```bash
python tools/benchmark.py plan --config configs/v4/protectai.json --environment ./u1-env
./u1-env/venv/bin/python tools/benchmark.py preflight --config configs/v4/protectai.json --environment ./u1-env --output runs/protectai-smoke
./u1-env/venv/bin/python tools/benchmark.py run --config configs/v4/protectai.json --environment ./u1-env --pilot --output runs/protectai-pilot
./u1-env/venv/bin/python tools/benchmark.py run --config configs/v4/protectai.json --environment ./u1-env --output runs/protectai-baseline
./u1-env/venv/bin/python tools/benchmark.py run --config configs/v4/protectai_compile.json --environment ./u1-env --baseline-ref runs/protectai-baseline --output runs/protectai-compile
python tools/benchmark.py report --output runs/protectai-compile
```

Для AccuKnox замените конфиг и venv на d3-вариант. Baseline не отключает нативный
fast path; наличие FLA в кандидате само по себе не доказывает изменение исполнения.

Готовые [baseline ZIP от 26 ноября](../results/u1_handoff/2026-11-26-new-families-baseline/README.md)
можно использовать как `--baseline-ref` при совпадении условий. В них сохранён
высокий A/A-envelope p95 у AccuKnox B1–B3 с batch=1 (33–47%): комплектность
`baseline_complete` не означает низкий шум или автоматическое одобрение кандидата.
Готовы [compile](../configs/v4/accuknox_compile.json),
[reference linear attention](../configs/v4/accuknox_reference.json) и
[FLA](../configs/v4/accuknox_fla.json). Скрытый override `D3_CUDNN_SDPA=1`
отклоняется: текущий контракт использует авторскую стандартную политику без cuDNN SDPA.
Preflight проверяет фактический dispatch. ORT классификатора проверяет все формы,
размещение узлов и `metadata-cpu-v1`; в измеряемой сессии профайлер выключен.
Качество подключается отдельно с явным mapping; успешный smoke не является quality-approval.

### Своя модель и нестандартный runtime

Для HF-классификации задайте weights/tokenizer с полными revisions, `preparation`
(`limit`, `text_selection`, `add_special_tokens`), `scoring` и `execution`.
`text_selection`: `assistant_else_user`, `user` либо `assistant`; отсутствие нужной
роли учитывается как неприменимость. `scoring.method`: `softmax_argmax` для любого
числа классов или `softmax_threshold` с двумя ожидаемыми `labels`, `positive_label`
и `threshold`. Метки сверяются с checkpoint; неизвестные параметры не игнорируются.

Пограничные примеры preflight синтетические: их формат выбирается через preparer
среди разрешённых конфигом user-only, assistant-only и user+assistant. Поэтому
assistant-only и pair-only не требуют добавлять user-only в контракт. Это не
преобразование корпуса: реальные сообщения, роли и контекст остаются неизменными.
Точная длина и отказ при превышении обязательны; ошибки подготовки не скрываются.

Для другого runtime остаётся [внешний Python-модуль](../examples/u1_classifier_extension.py)
с `create_preparer/create`, хешем и явными зависимостями. Общий
`u1_benchmark.family_adapter.FamilyAdapter` предоставляет индексацию, проверку
фактических IDs/masks, границ, размеры, precision и очистку. Подкласс предоставляет
реальный `forward_batch`, preparer и модуль, получающий входные тензоры;
дополнительный readout передаётся как `extra_tensors`. Нет наблюдаемых тензоров —
нет подтверждения проверки. Для нестандартного backend необходимы собственные
подготовка и доказательство исполнения, а не fallback на цикл одиночных запросов.
Автоматический подбор оптимизаций не входит в этот интерфейс.

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

Проверьте `data_ready`, `model_loads`, `preflight_stages`, `latency_stages`,
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

**GPU:** search запускает только candidate. Confirm необязателен и запускается
только явно, когда нужен отдельный парный эксперимент.

```bash
./u1-env/venv/bin/python tools/benchmark.py run --config configs/my-koala-candidate.json --environment ./u1-env --baseline-ref runs/my-koala-baseline --output runs/my-koala-search
./u1-env/venv/bin/python tools/benchmark.py run --config configs/my-koala-candidate.json --environment ./u1-env --mode confirm --output runs/my-koala-confirm
```

Изменённая сетка требует совместимого нового baseline; reference не пересчитывается
скрыто. Confirm не использует старые timing. Отдельная проверка мониторинга
не является условием положительного результата search или confirm.

### Уже есть baseline v4

Используйте ту же команду с `--baseline-ref`: каталог или ZIP распознаётся автоматически.
Не нужно конвертировать архив, выбирать версию или включать обход проверок.
До загрузки весов CLI сообщает совместимость и число загрузок; сообщения идут
в stderr, машинный JSON остаётся в stdout. Например, при разных GPU нужно выбрать
reference для текущей GPU, а не отключать проверку.

У v4 каждая серия выполнялась в свежем процессе, теперь модель сохраняется между
сериями. Это допустимое отличие явно отмечается: эффект относится ко всей новой
конфигурации/исполнению, а не только backend. Пики памяти также относятся к разным
lifecycle. Внутренние хеши архива проверяются; отличие общего хеша кода между
поддерживаемыми версиями само по себе не запрещает сравнение.

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
| baseline | Preflight; три основные серии baseline; затем A/A две пары ×30 |
| search | Только candidate: preflight и три основные серии по сохранённым планам reference; baseline/A/A не повторяются |
| confirm | Новый preflight обеих сторон; A/A две пары ×30; три A/B-пары с чередованием AB/BA |
| pilot | Preflight и одна основная серия/пара ×12; без A/A и без итогового pass |
| quality | Отдельные данные и сетка; один проход с неполным хвостом; без скоростных серий |

Перед каждой timed-стороной/серией 20 прогревочных батчей. Baseline/search используют
один процесс на ячейку: preflight, подготовка всех основных форм, серии, затем A/A
только у baseline. В confirm соседние стадии одного варианта объединяются без
перестановок AB/BA. Компиляция/экспорт/необходимые формы готовятся вне latency; eager не делает
дополнительный полный проход. Backend-proof выполняется в preflight, не заново в
каждой серии. Основная серия: `max(100, ceil(N/B))` полных батчей.

Search сверяет baseline-конфиг, входы, планы, совместимость протокола, GPU/VRAM, драйвер,
CUDA, пакеты, вычислительные настройки и метод памяти. Несовместимость — ошибка,
а не повод автоматически запускать baseline. Совпавшие параметры не доказывают
отсутствие дрейфа между сессиями. Search использует непарные блочные интервалы
из штатных серий обеих сторон и может получить `reference_pass`, без обязательного
confirm. Отдельной команды контроля baseline и скрытых повторных прогонов нет.

| Модели / режим | Загрузок раньше | Теперь без дополнительных профилей |
| --- | --- | --- |
| Decoder, baseline / search / confirm, 8 ячеек | 64 / 32 / 96 | 8 / 8 / 32 |
| Encoder, baseline / search / confirm, 2 ячейки | 16 / 8 / 24 | 2 / 2 / 8 |
| Quality, один вариант и три размера | 3 | 1 |

Это сокращение загрузок, не измеренное wall-clock ускорение. Число запросов не
уменьшается. Runtime/allocator-кэши сохраняются внутри процесса; ответы, GPU-входы
и KV между независимыми запросами не кешируются. Токенизация остаётся внутри таймера.
После ошибки worker не перезапускается автоматически: остаток его стадий `not_run`,
завершённые результаты сохранены, другие независимые workers могут продолжить работу.

## 5. Память, профили и статусы

Память снимается одновременно со скоростью. Один процесс nvidia-smi на worker,
запрошенный интервал 100 мс; загрузка, подготовка, прогрев и серии разделены окнами.
Setup учитывается один раз; peaks сбрасываются после прогрева без очистки allocator.
Пик относится к прогретому worker, не к независимой холодной загрузке. Выборочный пик процесса
может пропустить короткие всплески. Torch allocated/reserved и размер артефакта
показаны отдельно; Torch не заменяет память ORT. Ошибка sampler не скрывает latency.
Трёх дополнительных memory-проходов больше нет. `--profile-trace` включает отдельные
short/long-профили; без флага они не запускаются.

Исторический инструмент отдельной проверки sampler (не обязательная стадия запуска):

```bash
./u1-env/venv/bin/python tools/check_v4_monitoring.py --configs configs/v4/gliner.json configs/v4/open.json --environment ./u1-env --output runs/monitoring-check
```

Её `monitor_validation.json` можно указать в конфиге через `monitor_validation`.
Её наличие и применимость показываются информационно, без запрета `reference_pass`
или `confirm_pass`. Старые проверки не отменяются, но не доказывают свойства
изменённого sampler. Метод и настройки сбора у сравниваемых сторон должны совпадать.
При отличии хеша исходников прежняя проверка показывается как `historical_only`:
её `recorded_status` сохраняется, `current_implementation_validated=false`.

- `baseline_complete`: комплектный reference, не ускорение.
- `reference_pass`: критерии пройдены относительно сохранённого reference; не парное подтверждение.
- `exploratory`: исторический статус search в архивах v4, не пересматривается задним числом.
- `confirm_pass`: парное подтверждение в одной сессии, не независимая репликация.
- `pilot_complete / preflight_passed`: короткая проверка, не benchmark.
- `inconclusive / partial / incomplete`: шум, недостающие стадии или блокер; детали по ячейкам.
- `quality_complete`: оценка закончена; не означает «качество сохранено».
- `speed_status=below_target`: устойчивый эффект ниже порога либо замедление;
  для engineering/parameter итог сравнения `rejected`. Ошибки вердиктов также
  дают `rejected`, отдельно от статуса скорости.
- `quantization_measured`: сравнение записано, автоматического общего approval нет.

Инженерные изменения: строго >20% снижения p95, параметрические >10%, плюс прежние
проверки шума и вердиктов. «Быстрее во всей сетке» отдельно от достижения порога.
Квантование не получает автоматического quality-approval.
В JSON `comparison_design` различает `unpaired_reference` и `paired`.
Отметка всей сетки сохраняет это основание; search не становится парным
экспериментом из-за положительного результата. Разброс трёх штатных серий
не исключает систематического межсессионного сдвига.

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
Модель загружается один раз на вариант для всей quality-сетки. Перед каждым размером
восстанавливается RNG-состояние после загрузки; результаты/планы отдельные,
формы и backend проверяются заново. После ошибки остаток задач worker отмечается
not_run без автоматической перезагрузки или уменьшения batch.
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
[Текущая CPU-проверка](u1_handoff_readiness.md#переиспользование-baseline-и-модели-19-ноября-2026):
508 тестов без ошибок и пропусков; новый GPU lifecycle пока не проверен.
[Историческая проверка v4](../results/u1_handoff/2026-11-12-v4-release/validation.json):
490 CPU-тестов прошли в чистой копии без ошибок и пропусков.
Историческая [повторная индексация](../results/u1_handoff/2026-11-12-v4-cpu/validation.json)
подтвердила воспроизводимость индексов; корпус этим этапом не менялся.
[Верхние smoke-конфиги](../configs/v4/smoke) охватывают четыре encoder B1×256
и Open B1×64/B2×32/B3×16/B4×8.
[Новый Open BF16/native](../results/u1_handoff/2026-11-12-v4-native-open/README.md)
прошёл все четыре верхние ячейки. Остальные новые режимы этим не проверены.
Полные baseline/search/confirm GPU-серии в текущую приёмку не входят.
Переиспользование runtime новой редакции требует отдельной GPU-проверки владельцем;
старые результаты остаются валидными, но этой проверки не заменяют.
