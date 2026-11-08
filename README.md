# HiveTrace-AutoTune-Guardrails
Найти ускорения guardrail-моделей без скрытого падения качества на редких атаках, обфускации и неанглийских текстах в продукте

## Скелет CLI

Минимальная версия Python — **3.10**. Для разработки по умолчанию используется
Python 3.12 из `.python-version`. Зависимости и команды запуска управляются через
[uv](https://docs.astral.sh/uv/). GPU-бэкенды могут иметь дополнительные
ограничения по Python, CUDA и операционной системе.

```bash
uv sync --all-extras
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

Для локального каталога модели достаточно базовой установки.
Дополнение `hub` позволяет читать метаданные Hugging Face.

```bash
uv sync --extra hub
uv run guardrail-tuner inspect KoalaAI/Text-Moderation
uv run guardrail-tuner plan KoalaAI/Text-Moderation
uv run guardrail-tuner plan ./local-checkpoint
```

`inspect` читает `config.json`. Для Hugging Face можно указать `--revision`.
На этом этапе CLI загружает только конфиг.

`plan` предлагает проверки по типу архитектуры из конфига. Среди кандидатов
есть vLLM, ONNX Runtime, `torch.compile` и TensorRT через ONNX. Для
`torch.compile` план предусматривает сравнение с PyTorch eager, а для TensorRT
сначала нужно проверить экспорт и запуск на CUDA. Статус
`unverified` означает, что совместимость нужно подтвердить запуском.
`needs_inference_contract` означает, что CLI пока нужен способ подготовить
вход и сравнить выход модели с эталоном.

Следующий этап — выполнять эти проверки, а затем измерять прошедшие варианты
на заданной нагрузке. Результаты текущего `plan` описывают кандидатов.

## Датасеты качества и скорости

Установите поддержку датасетов и авторизуйтесь в Hugging Face через
`hf auth login` или переменную окружения `HF_TOKEN`. Токен должен иметь
доступ на чтение приватных наборов.

```bash
uv sync --extra datasets
uv run guardrail-tuner datasets \
  --quality-hf my-org/quality --quality-split test --quality-revision main \
  --performance-hf my-org/performance --performance-split test --performance-revision main
```

Вместо каждого HF ID можно независимо передать `--quality-file` или
`--performance-file` с путём к JSON, JSONL, CSV или Parquet. Например:

```bash
uv run guardrail-tuner datasets \
  --quality-hf my-org/quality --quality-split test \
  --performance-file ./performance.jsonl
```

Для HF также доступны `--quality-config` и `--performance-config` для выбора
конфигурации датасета. По умолчанию используется split `train`. Команда
загружает данные в стандартный кэш Hugging Face и выводит количество строк,
названия колонок и точный commit SHA. Для локальных файлов выводится SHA-256.
Сами записи в отчёт не попадают.

ID и схема приватных наборов пока не известны, поэтому они передаются при
запуске. Команда проверяет загрузку обоих источников. Выполнение оценки качества
и замера скорости требует реализации инференса и сопоставления полей датасета.

Правила разработки и формат коммитов описаны в [CONTRIBUTING.md](CONTRIBUTING.md).
