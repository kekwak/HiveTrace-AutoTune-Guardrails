# HiveTrace AutoTune Guardrails

**CLI для настройки и запуска инференса guardrail-моделей.**

[Быстрый старт](#быстрый-старт) · [Руководство по CLI](docs/cli-guide.md) · [Разработка](CONTRIBUTING.md)

## Быстрый старт

**`plan → run --config → serve --config`**

Нужны Python 3.10+ и [uv](https://docs.astral.sh/uv/). Выполните из корня репозитория.

```bash
uv tool install '.[inference,inference-cpu]'

guardrail-tuner plan KoalaAI/Text-Moderation > plan.json

guardrail-tuner run --config plan.json \
  --text "The meeting starts at noon." > run.json

guardrail-tuner serve --config plan.json
```

| Команда | Результат |
| --- | --- |
| `plan` | Одна конфигурация по эвристикам для архитектуры и задачи |
| `run --config` | Инференс с настройками из конфига и вывод предсказаний |
| `serve --config` | Локальный API с теми же настройками |

Передайте текст через `--text` или stdin. Повторите `--text`, чтобы обработать
несколько входов за один вызов. Для поддерживаемых моделей Transformers
план задаёт батчинг для классификации или KV-кэш для генерации.

```bash
curl http://127.0.0.1:8000/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":["The meeting starts at noon.","Please review the invoice."]}'
```

## Дополнительные проверки

| Команда | Назначение |
| --- | --- |
| `inspect`, `datasets` | Прочитать конфиг модели и проверить источники данных |
| `benchmark`, `optimize` | Сравнить PyTorch и ONNX Runtime для классификации |
| `tune-model`, `check-generation`, `compare-app` | Проверить компиляцию, кэш генерации или прикладные выходы |

Установка для CUDA, параметры сравнения и пример на трёх моделях доступны
в [руководстве по CLI](docs/cli-guide.md).
