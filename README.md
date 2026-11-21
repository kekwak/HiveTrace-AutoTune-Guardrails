# HiveTrace AutoTune Guardrails

**CLI для сравнения оптимизаций инференса guardrail-моделей на ваших данных.**

[Быстрый старт](#быстрый-старт) · [Руководство по CLI](docs/cli-guide.md) · [Совместимость с vLLM](docs/vllm-compatibility.md) · [Разработка](CONTRIBUTING.md)

## Быстрый старт

**`plan → run --config → serve --config`**

Нужны Python 3.10+ и [uv](https://docs.astral.sh/uv/). Выполните из корня репозитория.

```bash
uv tool install '.[inference,inference-cpu]'

guardrail-tuner plan KoalaAI/Text-Moderation > plan.json

guardrail-tuner run --config plan.json \
  --quality-file quality.jsonl --performance-file speed.jsonl > run.json

guardrail-tuner serve --config run.json
```

| Команда | Результат |
| --- | --- |
| `plan` | JSON-конфиг с оптимизацией по архитектуре чекпойнта |
| `run --config` | Сравнение с базовым режимом, проверка выходов и замер скорости после прогрева |
| `serve --config` | Локальный API с настройками из отчёта |

`quality.jsonl` и `speed.jsonl` — отдельные наборы со строковым полем `text`.
Первый проверяет совпадение выходов, второй задаёт нагрузку для замера.
Для поддерживаемых моделей Transformers план предлагает батчинг для классификации
или KV-кэш для генерации. Для KoalaAI пример проверяет логиты и argmax.
Для проверки решений с порогами используйте `benchmark --output-mode multilabel`.

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
| `probe-vllm` | Проверить запуск vLLM и сравнить выходы с Transformers |
| `tune-model`, `check-generation`, `compare-app` | Проверить компиляцию, кэш генерации или прикладные выходы |

Установка для CUDA, параметры сравнения и пример на трёх моделях доступны
в [руководстве по CLI](docs/cli-guide.md).
