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
| [HaloGuard1 Gen 0.8B](https://huggingface.co/astroware/HaloGuard1-Gen-0.8B/tree/85ff68dd063a7e58487a4251824fd36a20977d5c) | Colab Tesla T4, vLLM 0.34.0, FP16, eager, `language_model_only=True`: получены 2 токена. Проверен текстовый инференс напрямую в vLLM |

## Кандидаты на штатный запуск

| Чекпойнт | Почему |
| --- | --- |
| [ShieldGemma 2B](https://huggingface.co/google/shieldgemma-2b), [9B](https://huggingface.co/google/shieldgemma-9b) | `Gemma2ForCausalLM` подтверждён оригинальными конфигами обеих моделей. Для модерации важны шаблон и оценки токенов Yes/No |
| [WildGuard](https://huggingface.co/allenai/wildguard) | `MistralForCausalLM` подтверждён оригинальным конфигом |
| [HaloGuard1 Gen 4B](https://huggingface.co/astroware/HaloGuard1-Gen-4B) | Нативный класс `Qwen3_5ForConditionalGeneration` подтверждён оригинальным конфигом. Текстовый инференс проверяется |
| [CodeIntegrity PromptGuard](https://huggingface.co/codeintegrity-ai/promptguard) | vLLM 0.34.0 распознал `ModernBertForSequenceClassification` нативно. В Colab на T4 запуск остановился на инициализации EngineCore. Классификация пока не подтверждена |

## Нативный класс не найден в реестре vLLM 0.34.0

| Чекпойнт | Почему |
| --- | --- |
| [KoalaAI Text Moderation](https://huggingface.co/KoalaAI/Text-Moderation/blob/main/config.json) | В реестре нет `DebertaForSequenceClassification` |
| [ProtectAI DeBERTa v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2/blob/main/config.json), [deepset DeBERTa](https://huggingface.co/deepset/deberta-v3-base-injection/blob/main/config.json), Prompt Guard 2 [22M](https://huggingface.co/gravitee-io/Llama-Prompt-Guard-2-22M-onnx/blob/main/config.json) и [86M](https://huggingface.co/sinatras/Llama-Prompt-Guard-2-86M-ONNX/blob/main/config.json) | В реестре нет `DebertaV2ForSequenceClassification`. Для Prompt Guard класс взят из публичных копий |
| [Octavio Santana DeBERTa](https://huggingface.co/Octavio-Santana/deberta-v3-base-prompt-injection-detection) | Оригинальный `config.json` подтверждает `DebertaV2ForSequenceClassification`. Ревизия `c9f71de4e20cc011dbd9b458179bc68eee2cf4c9` |

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
| [Lunaris Guard](https://huggingface.co/auren-research/lunaris-guard/blob/7616e3df98e58d5b4fbcca20470acdae4a9c0bda/modeling_lunaris_guard.py) | Есть `auto_map.AutoModel`. Модель возвращает отдельные `injection_logits` и `safety_logits` из двух голов. Загрузка через Transformers backend остановилась при инициализации на meta device |
| [Nandi Mini 150M](https://huggingface.co/FrontiersMind/Nandi-Mini-150M-GuardRails/tree/e77ee070b9f332b3314fd28473737efd2bf94677) | Есть `auto_map.AutoModel`. На T4 vLLM 0.34.0 выбрал `TransformersForCausalLM`, но остановился на загрузке весов: в обёртке нет параметра `lm_head_proj` |
| [Nandi Mini 600M](https://huggingface.co/FrontiersMind/Nandi-Mini-600M-GuardRails/tree/9fb12b1407e5250bb7c021500b5e0bec2a8db7e3) | Есть `auto_map.AutoModel`. Генерация через Transformers backend подтверждена в GPU-прогоне выше. Текущий CLI этот backend не выбирает |
| [AccuKnox Guard 1.1](https://huggingface.co/Accuknoxtechnologies/accuknox-guard-1.1/blob/e4e43d89a1f2c0ae2025b50ec884d8de32d2cde3/modeling_d3.py) | Есть `auto_map.AutoModel → D3Model`. Загрузка использует собственный D3 runtime. Transformers backend остановился: в `D3Model.forward` нет аргумента `input_ids` |

## Пакетная проверка Colab

Целевое окружение — Tesla T4 и vLLM 0.34.0. Пробы использовали FP16, eager,
`max_model_len=128` и один нейтральный текст. Генерация запрашивала четыре
токена, классификация использовала pooling. Это проверка запуска. Замеры
скорости, сравнение с исходным рантаймом и качество модерации не проводились.

| Чекпойнт | Ревизия | Результат |
| --- | --- | --- |
| ShieldGemma 2B | `d1dffc9c8c9237a90aab09c61383791e718ef9e8` | Оригинальный конфиг прочитан с авторизацией. В Colab CLI без HF-токена получен 401 до загрузки |
| ShieldGemma 9B | `b8b636016df4540721a098c7aab91c97ec6ee508` | Оригинальный конфиг прочитан с авторизацией. В Colab CLI без HF-токена получен 401 до загрузки |
| WildGuard | `cbba4823f3e8020e5a74a5e29bf85072def6f2ff` | Оригинальный конфиг прочитан с авторизацией. В Colab CLI без HF-токена получен 401 до загрузки |
| CodeIntegrity PromptGuard | `c66a9b35c82079773c3119f48caf19be18c6ca22` | Предыдущий авторизованный прогон в Colab распознал класс, но завершился ошибкой EngineCore. В пакетной сессии без HF-токена получен 401 |
| HaloGuard1 Gen 0.8B | `85ff68dd063a7e58487a4251824fd36a20977d5c` | Нативный запуск распознал класс, затем остановился на загрузке image processor. Повтор с `language_model_only=True` и 15-минутным таймаутом вернул 2 токена |
| HaloGuard1 Gen 4B | `f157c1f8e81a5d13456bc659059461e2622cd36b` | Обычный запуск остановился на image processor. В длинном text-only прогоне наблюдалась загрузка FP16-весов и настройка KV-кэша. Генерация на момент обновления отчёта не подтверждена |
| KoalaAI Text Moderation | `838ade0edb66dcffc5532d08ff6ed5c899abfb5c` | Transformers backend: `ModelConfig` отклонил реализацию как несовместимую с vLLM до загрузки весов |
| ProtectAI DeBERTa v2 | `90c9989b1a342275dd0d1a95aad283c04e075671` | Transformers backend: `ModelConfig` отклонил реализацию как несовместимую с vLLM до загрузки весов |
| deepset DeBERTa | `80dda00d0b0d9a03917a7685e2ddbcd28e04dbb1` | Transformers backend: ошибка валидации `ModelConfig` до загрузки весов |
| Octavio Santana DeBERTa | `c9f71de4e20cc011dbd9b458179bc68eee2cf4c9` | Transformers backend: ошибка валидации `ModelConfig` до загрузки весов |
| Meta Prompt Guard 2 22M | — | В Colab CLI без HF-токена получен 401, отдельный авторизованный запрос вернул 403 |
| Meta Prompt Guard 2 86M | — | В Colab CLI без HF-токена получен 401, отдельный авторизованный запрос вернул 403 |
| GLiNER Guard Uniencoder | `5644cdb7c953463fa3a370e76617c03c2dd29f92` | В опубликованном конфиге нет `auto_map.AutoModel`. Проверка остановлена на конфиге |
| GLiNER Guard Omni | `763f8aa9a7714dd2c3264f3751acdd5d854c5ac4` | В опубликованном конфиге нет `auto_map.AutoModel`. Проверка остановлена на конфиге |
| GLiGuard 300M | `7776816b54581f37b90dd2de12dbe13123229a9b` | В опубликованном конфиге нет `auto_map.AutoModel`. Проверка остановлена на конфиге |
| OPIR Multitask Large | `69bb27407d66eab4797049d36ba75eef3579335b` | В опубликованном конфиге нет `auto_map.AutoModel`. Проверка остановлена на конфиге |
| OPIR Multitask Multilang | `1c1d66bd22e0ba93a435c710a9d9499be292cd65` | В опубликованном конфиге нет `auto_map.AutoModel`. Проверка остановлена на конфиге |
| Laya | `fd4f9dbcf4fde222fd43174bdd4c3b08b789e4e1` | В опубликованном конфиге нет `auto_map.AutoModel`. Проверка остановлена на конфиге |
| Lunaris Guard | `7616e3df98e58d5b4fbcca20470acdae4a9c0bda` | vLLM разрешил класс как `TransformersEmbeddingModel`. В EngineCore вызов `from_pretrained` внутри модели завершился ошибкой при создании на meta device. Обе головы не проверены |
| AccuKnox Guard 1.1 | `e4e43d89a1f2c0ae2025b50ec884d8de32d2cde3` | Transformers backend остановился на проверке `D3Model.forward`: нет аргумента `input_ids` |
| Nandi Mini 150M | `e77ee070b9f332b3314fd28473737efd2bf94677` | Предыдущая проба Transformers backend остановилась на загрузке `lm_head_proj` |
| Nandi Mini 600M | `9fb12b1407e5250bb7c021500b5e0bec2a8db7e3` | Предыдущая проба Transformers backend вернула 4 токена |
| HiveTraceGuard-Pro | `ed008276c96253e14885e6ade3b7b6d58daa2480` | Предыдущий нативный прогон `plan → run → serve` вернул совпадающие токены |

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
