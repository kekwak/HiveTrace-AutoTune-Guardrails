# Совместимость моделей с vLLM

Проверка от 21 ноября 2026 года основана на конфигурациях и карточках
чекпойнтов, [реестре vLLM][registry] и [списке поддерживаемых моделей][supported].
Статус относится к запуску полного чекпойнта в штатном vLLM без собственного
адаптера. Статическую проверку дополняют GPU-прогоны, где они указаны.

| Статус | Значение |
| --- | --- |
| Проверено на GPU | Чекпойнт загрузился и выполнил инференс в указанном окружении |
| Кандидат | Архитектура поддерживается. Загрузку и выходы нужно подтвердить на целевой версии vLLM |
| Нет штатного пути | В реестре нет нужного класса либо чекпойнт использует собственный интерфейс |
| Не установлено | Недостаточно сведений о конфиге или поставке весов |

## Проверено на GPU

| Чекпойнт | Результат |
| --- | --- |
| [HiveTraceGuard-Pro](https://huggingface.co/hivetrace/HiveTraceGuard-Pro) | В Colab на Tesla T4 с vLLM 0.34.0, PyTorch 2.14.0+cu130 и Python 3.13.16 прошёл `plan → run → serve`. План включил FP16 и prefix caching. На одном входе CLI и `POST /infer` вернули одинаковые 8 токенов и текст. Ревизия модели `ed008276c96253e14885e6ade3b7b6d58daa2480`. Скорость, качество модерации и паритет с Transformers в этом прогоне не измерялись |

## Кандидаты на штатный запуск

| Чекпойнт | Почему |
| --- | --- |
| [ShieldGemma 2B](https://huggingface.co/google/shieldgemma-2b), [9B](https://huggingface.co/google/shieldgemma-9b) | Gemma 2, `Gemma2ForCausalLM`. Класс проверен по публичным копиям. Для модерации важны шаблон и оценки токенов Yes/No |
| [WildGuard](https://huggingface.co/allenai/wildguard) | Карточка подтверждает Mistral 7B v0.3. Оригинальный конфиг в этой проверке недоступен |
| [HaloGuard1 Gen 0.8B](https://huggingface.co/astroware/HaloGuard1-Gen-0.8B/blob/main/config.json), [4B](https://huggingface.co/astroware/HaloGuard1-Gen-4B) | `Qwen3_5ForConditionalGeneration` подтверждён конфигом 0.8B и карточкой 4B. Нужен релиз vLLM с поддержкой Qwen3.5 |
| [CodeIntegrity PromptGuard](https://huggingface.co/codeintegrity-ai/promptguard) | vLLM 0.34.0 распознал `ModernBertForSequenceClassification` нативно. В Colab на T4 запуск остановился на инициализации EngineCore. Классификация пока не подтверждена |

## Нет штатного пути по проверенному реестру

| Чекпойнт | Почему |
| --- | --- |
| [KoalaAI Text Moderation](https://huggingface.co/KoalaAI/Text-Moderation/blob/main/config.json) | В реестре нет `DebertaForSequenceClassification` |
| [ProtectAI DeBERTa v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2/blob/main/config.json), [deepset DeBERTa](https://huggingface.co/deepset/deberta-v3-base-injection/blob/main/config.json), Prompt Guard 2 [22M](https://huggingface.co/gravitee-io/Llama-Prompt-Guard-2-22M-onnx/blob/main/config.json) и [86M](https://huggingface.co/sinatras/Llama-Prompt-Guard-2-86M-ONNX/blob/main/config.json) | В реестре нет `DebertaV2ForSequenceClassification`. Для Prompt Guard класс взят из публичных копий |
| [GLiNER Guard Uniencoder](https://huggingface.co/hivetrace/gliner-guard-uniencoder), [Omni](https://huggingface.co/hivetrace/gliner-guard-omni), [GLiGuard 300M](https://huggingface.co/fastino/gliguard-LLMGuardrails-300M) | GLiNER2 extractor со своими головами и схемой вывода. Поддержка базового энкодера сама по себе не покрывает этот интерфейс |
| [OPIR Multitask Large](https://huggingface.co/knowledgator/opir-multitask-large-v1.0/blob/main/config.json), [Multilang](https://huggingface.co/knowledgator/opir-multitask-multilang-v1.0/blob/main/config.json) | Собственный класс `GLiClassModel` |
| [Laya](https://huggingface.co/convaiinnovations/laya/blob/main/config.json) | Обёртка `LayaTypedDecisions` поверх ModernBERT |
| [Lunaris Guard](https://huggingface.co/auren-research/lunaris-guard/blob/main/config.json) | Собственный `model_type: lunaris_guard` и `auto_map` |
| [Nandi Mini 150M](https://huggingface.co/FrontiersMind/Nandi-Mini-150M-GuardRails), [600M](https://huggingface.co/FrontiersMind/Nandi-Mini-600M-GuardRails) | Карточки используют собственную реализацию с `trust_remote_code=True` |
| [AccuKnox Guard 1.1](https://huggingface.co/Accuknoxtechnologies/accuknox-guard-1.1) | Собственный классификатор и процессор поверх модели семейства Qwen3.5. Карточка использует `trust_remote_code=True` |

## Не установлено

| Чекпойнт | Что проверить |
| --- | --- |
| [Octavio Santana DeBERTa](https://huggingface.co/Octavio-Santana/deberta-v3-base-prompt-injection-detection) | Карточка указывает DeBERTa и PEFT. Уточнить базовую модель и формат весов. Для стандартной головы DeBERTa действует ограничение реестра выше |

## Проверка на целевой машине

Для отдельного сравнения с Transformers на Linux с CUDA и установленным vLLM:

```bash
guardrail-tuner probe-vllm hivetrace/HiveTraceGuard-Pro \
  --revision MODEL_COMMIT --quality-file quality.jsonl --manifest vllm-run.json
```

Замените `MODEL_COMMIT` на ревизию модели. В колонке `text` используйте готовые
входы по шаблону модели. `probe-vllm` сверяет токены генерации или, в режиме
`--task classification`, вероятности и решения. Для оценки прикладного
результата отдельно проверяйте scores, пороги и разбор ответа через `compare-app`.

Проба загружает эталон через Transformers без `trust_remote_code` и использует
FP32. Ей нужны совместимая версия Transformers и достаточная VRAM. Например,
карточка HaloGuard указывает Transformers 5.7, а extra проекта ограничен
`transformers<5`. Успех загрузки vLLM и успех сравнительной пробы проверяются
отдельно.

[registry]: https://docs.vllm.ai/en/stable/api/vllm/model_executor/models/registry/
[supported]: https://docs.vllm.ai/en/stable/models/supported_models/
