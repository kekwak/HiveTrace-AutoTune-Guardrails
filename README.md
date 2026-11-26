# HiveTrace AutoTune Guardrails

**CLI для настройки и запуска инференса guardrail-моделей.**

[Быстрый старт](#быстрый-старт) · [Демонстрация CLI](#демонстрация-cli) · [Руководство по CLI](docs/cli-guide.md) · [Совместимость с vLLM](docs/vllm-compatibility.md) · [Разработка](CONTRIBUTING.md)

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

## Демонстрация CLI

Две модели из исходного списка и внешняя модель семейства BART проходят один
сценарий на CPU. В GIF показаны сокращённые выводы реальных команд на версии
`e146df0`. Поле `rules.applied` подтверждает, что правило `c12` применилось и
на ProtectAI, и на BART.

![Три запуска CLI на CPU. Для каждой модели plan сохраняет конфигурацию, а run показывает применённые правила и метки.](docs/assets/cli-demo.gif)

Для живого показа выполните сценарий ниже на версии `e146df0` или на ветке,
которая включает этот коммит. Прогоните его заранее, чтобы скачать веса моделей.
На том же компьютере команды можно повторить без сети, задав `HF_HUB_OFFLINE=1`.
Для сокращённого вывода нужен `jq`.

```bash
uv sync --locked --extra inference --extra inference-cpu

demo() {
  local model="$1" revision="$2" name="$3"
  uv run --locked --no-sync guardrail-tuner plan "$model" --revision "$revision" > "$name.plan.json" &&
  jq '.execution | {optimization, batch_size, rules}' "$name.plan.json" &&
  uv run --locked --no-sync guardrail-tuner run --config "$name.plan.json" \
    --text 'The meeting starts at noon.' --text 'Please review the invoice.' > "$name.run.json" &&
  jq '{applied: .rules.applied, parity: .rules.parity, labels: [.outputs[].label]}' "$name.run.json"
}

demo KoalaAI/Text-Moderation 838ade0edb66dcffc5532d08ff6ed5c899abfb5c koala
demo protectai/deberta-v3-base-prompt-injection-v2 90c9989b1a342275dd0d1a95aad283c04e075671 protectai
demo facebook/bart-large-mnli d7645e127eaf1aefc7862fd59a17a5aa8558b8ce bart
```

## Запуск через vLLM

В CUDA-окружении с установленным vLLM:

```bash
guardrail-tuner plan hivetrace/HiveTraceGuard-Pro --device cuda > plan.json
guardrail-tuner run --config plan.json --text "Please review the invoice."
guardrail-tuner serve --config plan.json
```

Для `ForCausalLM` из реестра установленного vLLM план выбирает штатный
vLLM с prefix caching и FP16. Для остальных поддерживаемых сценариев
используется Transformers. Проверить выходы отдельно можно через `probe-vllm`.

## Дополнительные проверки

| Команда | Назначение |
| --- | --- |
| `inspect`, `datasets` | Прочитать конфиг модели и проверить источники данных |
| `benchmark`, `optimize` | Сравнить PyTorch и ONNX Runtime для классификации |
| `probe-vllm` | Проверить запуск vLLM и сравнить выходы с Transformers |
| `tune-model`, `check-generation`, `compare-app` | Проверить компиляцию, кэш генерации или прикладные выходы |

Установка для CUDA, параметры сравнения и пример на трёх моделях доступны
в [руководстве по CLI](docs/cli-guide.md).
