# Совместимость моделей с vLLM

Проверка от 21 ноября 2026 года основана на конфигурациях и карточках
чекпойнтов, [реестре vLLM][registry] и [списке поддерживаемых моделей][supported].
Отчёт различает нативную поддержку полного чекпойнта, совместимость с
Transformers backend vLLM и доступность в текущем CLI. GPU-прогоны указаны
отдельно.

| Статус | Значение |
| --- | --- |
| Проверено на GPU | Чекпойнт загрузился и выполнил инференс в указанном окружении |
| Кандидат | Архитектура поддерживается. Загрузку и выходы нужно подтвердить на целевой версии vLLM |
| Нативный класс не найден | Прямой запуск через нативный реестр не подтверждён. Transformers backend проверяется отдельно |
| Собственный интерфейс | Возможность запуска полной модели и сохранение её выходов проверяются отдельно |
| Не установлено | Недостаточно сведений о конфиге или поставке весов |

## Проверено на GPU

| Чекпойнт | Результат |
| --- | --- |
| [HiveTraceGuard-Pro](https://huggingface.co/hivetrace/HiveTraceGuard-Pro) | В Colab на Tesla T4 с vLLM 0.34.0, PyTorch 2.14.0+cu130 и Python 3.13.16 прошёл `plan → run → serve`. План включил FP16 и prefix caching. На одном входе CLI и `POST /infer` вернули одинаковые 8 токенов и текст. Ревизия модели `ed008276c96253e14885e6ade3b7b6d58daa2480`. Скорость, качество модерации и паритет с Transformers в этом прогоне не измерялись |
| [Nandi Mini 600M](https://huggingface.co/FrontiersMind/Nandi-Mini-600M-GuardRails/tree/9fb12b1407e5250bb7c021500b5e0bec2a8db7e3) | В Colab на Tesla T4 с vLLM 0.34.0 загрузился через `model_impl="transformers"`, `trust_remote_code=True`, FP16 и eager. На одном входе сгенерировал 4 токена. Паритет с эталоном, качество и скорость не проверялись |

## Кандидаты на штатный запуск

| Чекпойнт | Почему |
| --- | --- |
| [ShieldGemma 2B](https://huggingface.co/google/shieldgemma-2b), [9B](https://huggingface.co/google/shieldgemma-9b) | Gemma 2, `Gemma2ForCausalLM`. Класс проверен по публичным копиям. Для модерации важны шаблон и оценки токенов Yes/No |
| [WildGuard](https://huggingface.co/allenai/wildguard) | Карточка подтверждает Mistral 7B v0.3. Оригинальный конфиг в этой проверке недоступен |
| [HaloGuard1 Gen 0.8B](https://huggingface.co/astroware/HaloGuard1-Gen-0.8B/blob/main/config.json), [4B](https://huggingface.co/astroware/HaloGuard1-Gen-4B) | `Qwen3_5ForConditionalGeneration` подтверждён конфигом 0.8B и карточкой 4B. Нужен релиз vLLM с поддержкой Qwen3.5 |
| [CodeIntegrity PromptGuard](https://huggingface.co/codeintegrity-ai/promptguard) | vLLM 0.34.0 распознал `ModernBertForSequenceClassification` нативно. В Colab на T4 запуск остановился на инициализации EngineCore. Классификация пока не подтверждена |

## Нативный класс не найден в реестре vLLM 0.34.0

| Чекпойнт | Почему |
| --- | --- |
| [KoalaAI Text Moderation](https://huggingface.co/KoalaAI/Text-Moderation/blob/main/config.json) | В реестре нет `DebertaForSequenceClassification` |
| [ProtectAI DeBERTa v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2/blob/main/config.json), [deepset DeBERTa](https://huggingface.co/deepset/deberta-v3-base-injection/blob/main/config.json), Prompt Guard 2 [22M](https://huggingface.co/gravitee-io/Llama-Prompt-Guard-2-22M-onnx/blob/main/config.json) и [86M](https://huggingface.co/sinatras/Llama-Prompt-Guard-2-86M-ONNX/blob/main/config.json) | В реестре нет `DebertaV2ForSequenceClassification`. Для Prompt Guard класс взят из публичных копий |

## Чекпойнты с собственным интерфейсом

Текущий `plan → run → serve` не создаёт исполняемый план для чекпойнта с
`auto_map`. Свой класс можно загрузить через [Transformers backend vLLM][custom]
при совместимой реализации и `auto_map.AutoModel` в конфиге. Для загрузки такого
кода с Hugging Face нужен `trust_remote_code=True`. Наличие `auto_map.AutoModel`
подтверждает точку входа, а совместимость проверяется запуском. Для моделей с
собственными головами требуется проверить, что доступны их прикладные выходы.

| Чекпойнт | Что проверено |
| --- | --- |
| [GLiNER Guard Uniencoder](https://huggingface.co/hivetrace/gliner-guard-uniencoder), [Omni](https://huggingface.co/hivetrace/gliner-guard-omni), [GLiGuard 300M](https://huggingface.co/fastino/gliguard-LLMGuardrails-300M) | В корневых конфигах есть `model_type: extractor`, поля `auto_map.AutoModel` нет. Выход формирует GLiNER2 extractor со своими головами. Прямой путь через опубликованный конфиг не найден |
| [OPIR Multitask Large](https://huggingface.co/knowledgator/opir-multitask-large-v1.0/blob/main/config.json), [Multilang](https://huggingface.co/knowledgator/opir-multitask-multilang-v1.0/blob/main/config.json) | В конфигах указан `GLiClassModel`, поля `auto_map.AutoModel` нет. Прямой путь через опубликованный конфиг не найден |
| [Laya](https://huggingface.co/convaiinnovations/laya/blob/main/config.json) | В корневом конфиге указан `LayaTypedDecisions`, поля `auto_map.AutoModel` нет. Совместимый ModernBERT backbone сам по себе не возвращает решения Laya |
| [Lunaris Guard](https://huggingface.co/auren-research/lunaris-guard/blob/7616e3df98e58d5b4fbcca20470acdae4a9c0bda/modeling_lunaris_guard.py) | Есть `auto_map.AutoModel`. Модель возвращает отдельные `injection_logits` и `safety_logits` из двух голов. Сохранение обоих выходов через vLLM не проверено |
| [Nandi Mini 150M](https://huggingface.co/FrontiersMind/Nandi-Mini-150M-GuardRails/tree/e77ee070b9f332b3314fd28473737efd2bf94677) | Есть `auto_map.AutoModel`. На T4 vLLM 0.34.0 выбрал `TransformersForCausalLM`, но остановился на загрузке весов: в обёртке нет параметра `lm_head_proj` |
| [Nandi Mini 600M](https://huggingface.co/FrontiersMind/Nandi-Mini-600M-GuardRails/tree/9fb12b1407e5250bb7c021500b5e0bec2a8db7e3) | Есть `auto_map.AutoModel`. Генерация через Transformers backend подтверждена в GPU-прогоне выше. Текущий CLI этот backend не выбирает |
| [AccuKnox Guard 1.1](https://huggingface.co/Accuknoxtechnologies/accuknox-guard-1.1/blob/e4e43d89a1f2c0ae2025b50ec884d8de32d2cde3/modeling_d3.py) | Есть `auto_map.AutoModel → D3Model`. Загрузка использует собственный D3 runtime. Сохранение выхода полного классификатора через vLLM не проверено |

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

[registry]: https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/model_executor/models/registry.py
[supported]: https://docs.vllm.ai/en/v0.34.0/models/supported_models/
[custom]: https://docs.vllm.ai/en/v0.34.0/models/supported_models/#custom-models
