# HiveTrace-AutoTune-Guardrails
Найти ускорения guardrail-моделей без скрытого падения качества на редких атаках, обфускации и неанглийских текстах в продукте

## Установка CLI

Минимальная версия Python — **3.10**. Из каталога репозитория установите CLI
через [uv](https://docs.astral.sh/uv/) и запускайте его напрямую.

Для CPU:

```bash
uv tool install '.[inference,inference-cpu]'
guardrail-tuner --help
```

Для NVIDIA CUDA:

```bash
uv tool install '.[inference,inference-cuda]'
guardrail-tuner --help
```

CPU и CUDA варианты устанавливаются отдельно. Для смены варианта используйте
`uv tool install --force` с нужными extras. Если команда после установки не
найдена, выполните `uv tool update-shell` и перезапустите терминал. Установка
инструмента разрешает версии зависимостей отдельно от проектного `uv.lock`.

```bash
guardrail-tuner inspect KoalaAI/Text-Moderation
guardrail-tuner plan KoalaAI/Text-Moderation
guardrail-tuner plan ./local-checkpoint
```

Для разработки используйте команды и выбор окружения из
[CONTRIBUTING.md](CONTRIBUTING.md).

`inspect` читает `config.json`. Для Hugging Face можно указать `--revision`.
На этом этапе CLI загружает только конфиг.

`plan` предлагает проверки по типу архитектуры из конфига. Среди кандидатов
есть vLLM, ONNX Runtime, `torch.compile` и TensorRT через ONNX. Для
`torch.compile` план предусматривает сравнение с PyTorch eager, а для TensorRT
сначала нужно проверить экспорт и запуск на CUDA. Статус
`unverified` означает, что совместимость нужно подтвердить запуском.
`needs_inference_contract` означает, что CLI пока нужен способ подготовить
вход и сравнить выход модели с эталоном.

Результаты `plan` описывают кандидатов. Команды `benchmark` и `optimize`
сравнивают PyTorch и ONNX Runtime для sequence classification.

## Как работает CLI

```mermaid
flowchart LR
    A["Выберите модель<br/>и CPU или CUDA"]
    B["Укажите два датасета<br/>качество и скорость"]
    C["Запустите benchmark<br/>PyTorch и ONNX Runtime"]
    D["Получите отчёт<br/>совпадение ответов и задержка"]

    A --> B --> C --> D
```

Команда `benchmark` загружает модель и датасеты, готовит ONNX и сравнивает
результаты с PyTorch на наборе качества. Затем она замеряет оба варианта на
наборе скорости и выводит JSON-отчёт. При расхождении сверх допуска команда
возвращает код 1.

Перед запуском можно отдельно изучить модель через `inspect`, посмотреть
непроверенные варианты через `plan` и проверить загрузку наборов через
`datasets`. Пример полного прогона находится в разделе
[«Первый прогон классификатора»](#первый-прогон-классификатора).

Сейчас `benchmark` поддерживает sequence classification и последовательные
запросы с batch size 1. Он сравнивает выходы с PyTorch и может дополнительно
оценить совпадение с размеченными данными.

## Датасеты качества и скорости

Установите поддержку датасетов и авторизуйтесь в Hugging Face через
`hf auth login` или переменную окружения `HF_TOKEN`. Токен должен иметь
доступ на чтение приватных наборов.

```bash
guardrail-tuner datasets \
  --quality-hf my-org/quality --quality-split test --quality-revision main \
  --performance-hf my-org/performance --performance-split test --performance-revision main
```

Вместо каждого HF ID можно независимо передать `--quality-file` или
`--performance-file` с путём к JSON, JSONL, CSV или Parquet. Например:

```bash
guardrail-tuner datasets \
  --quality-hf my-org/quality --quality-split test \
  --performance-file ./performance.jsonl
```

Для HF также доступны `--quality-config` и `--performance-config` для выбора
конфигурации датасета. По умолчанию используется split `train`. Команда
загружает данные в стандартный кэш Hugging Face и выводит количество строк,
названия колонок и точный commit SHA. Для локальных файлов выводится SHA-256.
Сами записи в отчёт не попадают.

ID и схема приватных наборов задаются при запуске. Команда `datasets`
проверяет загрузку источников, а `benchmark` использует их для сравнения
результатов и замера скорости.

Правила разработки и формат коммитов описаны в [CONTRIBUTING.md](CONTRIBUTING.md).

## Первый прогон классификатора

Команда `benchmark` поддерживает модели Transformers для sequence classification.
Два датасета должны содержать строковую колонку с текстом. Для Hugging Face
наборов вместо файлов доступны те же параметры `--quality-hf`,
`--performance-hf`, `--*-split`, `--*-revision` и `--*-config`, что и у
команды `datasets`.

```bash
guardrail-tuner benchmark KoalaAI/Text-Moderation \
  --device cpu --output-mode multilabel --threshold 0.5 \
  --quality-file ./quality.jsonl --performance-file ./speed.jsonl
```

Для CUDA установите совместимые драйвер и библиотеки CUDA/cuDNN, затем выберите
GPU-версию ONNX Runtime:

```bash
guardrail-tuner benchmark MODEL_ID --device cuda \
  --output-mode multiclass --quality-file ./quality.jsonl --performance-file ./speed.jsonl
```

CPU и CUDA extras взаимоисключающие. Для CUDA версии PyTorch и ONNX Runtime
должны быть совместимы по CUDA
и cuDNN. CLI проверяет доступность CUDA и активного ONNX Runtime provider.
ONNX Runtime может размещать отдельные неподдерживаемые операции на CPU.

Укажите `--output-mode multiclass` для softmax и argmax,
`--output-mode multilabel` для sigmoid и общего порога `--threshold`, либо
`--output-mode logits` для сравнения только логитов. Выбор зависит от модели.
`--max-logit-error` задаёт допустимое максимальное абсолютное отклонение
(по умолчанию 0.001). При превышении или несовпадении решений команда
возвращает код 1 и выводит результаты в JSON на stdout. Для анализа по разметке
добавьте `--quality-label-column`, как описано ниже.

Замер последовательный, с batch size 1 и без имитации очереди. Он включает
токенизацию, инференс и перенос выхода на CPU. Число запросов равно количеству
строк датасета скорости, поэтому для устойчивого p95/p99 нужен достаточно
большой набор. Экспорт ONNX и загрузка моделей выполняются до замера.

Имя колонки по умолчанию — `text`. Общую замену задаёт `--text-column`,
а разные колонки для наборов — `--quality-text-column` и
`--performance-text-column`.

## Автоподбор конфигурации

```bash
guardrail-tuner optimize KoalaAI/Text-Moderation \
  --device cpu --output-mode multilabel --threshold 0.5 \
  --quality-file ./quality.jsonl --performance-file ./speed.jsonl \
  --output-dir ./results/run-1
```

Для CUDA выберите `--device cuda` и установите `inference-cuda` вместо
`inference-cpu`. Параметры датасетов и выходов совпадают с `benchmark`.
Каталог результатов должен быть пустым или отсутствовать.

Команда проверяет уровни оптимизации графа ONNX Runtime `all` и `extended`.
На CPU также сравнивает число intra-op потоков 0 (авто) и 1. Каждый вариант
проверяется на датасете качества и измеряется на наборе скорости. CLI выбирает
вариант с минимальным p95 среди прошедших проверку или исходный PyTorch, если он
быстрее. Он сохраняет `config.yaml` с параметрами и `report.md` с результатами
всех кандидатов. Если ни один прогон не завершился, создаётся только отчёт,
а команда возвращает код 1.

Конфигурация предназначена для воспроизведения настроек. CLI пока не запускает
сервер из `config.yaml` и не сохраняет экспортированный ONNX. Это сравнение
одной последовательной нагрузки. Для режима `logits` совпадение решений и
вероятностей не проверяется.

## Проверка по разметке

Для `benchmark` и `optimize` можно указать `--quality-label-column label`.
В режиме `multiclass` значение колонки — целочисленный индекс класса из выходов
модели. В режиме `multilabel` — массив из 0 и 1 в порядке выходов модели.
Пример для двух выходов:

```json
{"text": "пример", "label": [0, 1]}
```

Отчёт содержит число точных совпадений с разметкой и TP, FP, TN и FN для
каждого индекса класса, отдельно для PyTorch и ONNX Runtime. Для `multilabel`
точное совпадение требует совпадения всех меток записи. В выборе конфигурации
по-прежнему используются совпадение решений с PyTorch и допуск на logits.
Разметка помогает оценить оба варианта, но сама по себе не меняет критерий отбора.
