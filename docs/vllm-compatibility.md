# Совместимость моделей с vLLM

Проверка от 21 ноября 2026 года охватывает **24 чекпойнта**. GPU-прогоны
выполнялись в Colab на Tesla T4 с vLLM 0.34.0 и FP16. Прямые прогоны
использовали eager, один нейтральный текст и до четырёх новых токенов.
Успех означает, что модель загрузилась и вернула результат в указанном
режиме. Качество модерации, паритет с эталоном и ускорение не измерялись.

Полный чекпойнт запустился и вернул результат **4** | Запуск завершился ошибкой до результата **9** | Запуск не проверен из-за доступа **5** | Проверен только конфиг **6**

## Полный чекпойнт запустился и вернул результат (4)

| Модель | Ревизия | Наблюдение |
| --- | --- | --- |
| [HiveTraceGuard-Pro](https://huggingface.co/hivetrace/HiveTraceGuard-Pro) | `ed008276c96253e14885e6ade3b7b6d58daa2480` | Нативный vLLM. `plan → run → serve` с FP16 и prefix caching. CLI и `POST /infer` вернули одинаковые 8 токенов и текст. |
| [Nandi Mini 600M](https://huggingface.co/FrontiersMind/Nandi-Mini-600M-GuardRails) | `9fb12b1407e5250bb7c021500b5e0bec2a8db7e3` | Transformers backend с `trust_remote_code=True`. Прямой запуск vLLM вернул 4 токена. |
| [HaloGuard1 Gen 0.8B](https://huggingface.co/astroware/HaloGuard1-Gen-0.8B) | `85ff68dd063a7e58487a4251824fd36a20977d5c` | Нативный vLLM с `language_model_only=True`. Полный текстовый инференс вернул 2 токена. Обычный запуск остановился на image processor. |
| [HaloGuard1 Gen 4B](https://huggingface.co/astroware/HaloGuard1-Gen-4B) | `f157c1f8e81a5d13456bc659059461e2622cd36b` | Нативный vLLM с `language_model_only=True`. Прямой текстовый инференс вернул 2 токена. |

## Запуск завершился ошибкой до результата (9)

| Модель | Ревизия | Наблюдение |
| --- | --- | --- |
| [CodeIntegrity PromptGuard](https://huggingface.co/codeintegrity-ai/promptguard) | `c66a9b35c82079773c3119f48caf19be18c6ca22` | Нативный vLLM распознал `ModernBertForSequenceClassification`, затем `EngineCore` завершился ошибкой инициализации. Первичная причина не установлена. |
| [KoalaAI Text Moderation](https://huggingface.co/KoalaAI/Text-Moderation) | `838ade0edb66dcffc5532d08ff6ed5c899abfb5c` | Нативного класса нет. `ModelConfig` отклонил `DebertaForSequenceClassification` в Transformers backend. |
| [ProtectAI DeBERTa v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) | `90c9989b1a342275dd0d1a95aad283c04e075671` | Нативного класса нет. `ModelConfig` отклонил `DebertaV2ForSequenceClassification` в Transformers backend. |
| [deepset DeBERTa](https://huggingface.co/deepset/deberta-v3-base-injection) | `80dda00d0b0d9a03917a7685e2ddbcd28e04dbb1` | Нативного класса нет. Transformers backend остановился на валидации `ModelConfig` до загрузки весов. |
| [Octavio Santana DeBERTa](https://huggingface.co/Octavio-Santana/deberta-v3-base-prompt-injection-detection) | `c9f71de4e20cc011dbd9b458179bc68eee2cf4c9` | Оригинальный конфиг объявляет `DebertaV2ForSequenceClassification`. Transformers backend остановился на валидации `ModelConfig`. |
| [Lunaris Guard](https://huggingface.co/auren-research/lunaris-guard) | `7616e3df98e58d5b4fbcca20470acdae4a9c0bda` | Transformers backend выбрал `TransformersEmbeddingModel`. Вложенный `from_pretrained` упал на meta device. Обе головы остались непроверенными. |
| [AccuKnox Guard 1.1](https://huggingface.co/Accuknoxtechnologies/accuknox-guard-1.1) | `e4e43d89a1f2c0ae2025b50ec884d8de32d2cde3` | Transformers backend остановился на проверке `D3Model.forward`: нет аргумента `input_ids`. |
| [AccuKnox Guard 1.2](https://huggingface.co/Accuknoxtechnologies/accuknox-guard-1.2) | `824db9ea82ac832e9e01cfa4883149d461a77fbe` | На T4 vLLM 0.34.0 выбрал `TransformersMultiModalForSequenceClassification` (`model_impl="transformers"`, pooling, `trust_remote_code=True`). `auto_map.AutoModel` указывает на `D3Model`. EngineCore остановился при проверке `D3Model.forward`: нет аргумента `input_ids`. Выход полной модели не получен. |
| [Nandi Mini 150M](https://huggingface.co/FrontiersMind/Nandi-Mini-150M-GuardRails) | `e77ee070b9f332b3314fd28473737efd2bf94677` | Transformers backend выбрал `TransformersForCausalLM` и остановился на загрузке весов: в обёртке отсутствует `lm_head_proj`. |

## Запуск не проверен из-за доступа (5)

| Модель | Ревизия | Наблюдение |
| --- | --- | --- |
| [ShieldGemma 2B](https://huggingface.co/google/shieldgemma-2b) | `d1dffc9c8c9237a90aab09c61383791e718ef9e8` | С токеном подтверждены `Gemma2ForCausalLM` и доступ к весам. Colab CLI без токена получил 401 до загрузки. |
| [ShieldGemma 9B](https://huggingface.co/google/shieldgemma-9b) | `b8b636016df4540721a098c7aab91c97ec6ee508` | С токеном подтверждены `Gemma2ForCausalLM` и доступ к весам. Colab CLI без токена получил 401 до загрузки. |
| [WildGuard](https://huggingface.co/allenai/wildguard) | `cbba4823f3e8020e5a74a5e29bf85072def6f2ff` | С токеном подтверждены `MistralForCausalLM` и доступ к весам. Colab CLI без токена получил 401 до загрузки. |
| [Meta Prompt Guard 2 22M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M) | — | В Colab CLI получен 401. Авторизованный запрос тоже не получил доступ (403). В публичной копии конфига указан `DebertaV2ForSequenceClassification`. |
| [Meta Prompt Guard 2 86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) | — | В Colab CLI получен 401. Авторизованный запрос тоже не получил доступ (403). В публичной копии конфига указан `DebertaV2ForSequenceClassification`. |

## Проверен только конфиг (6)

| Модель | Ревизия | Наблюдение |
| --- | --- | --- |
| [GLiNER Guard Uniencoder](https://huggingface.co/hivetrace/gliner-guard-uniencoder) | `5644cdb7c953463fa3a370e76617c03c2dd29f92` | GLiNER2 extractor. В корневом конфиге нет `auto_map.AutoModel`. |
| [GLiNER Guard Omni](https://huggingface.co/hivetrace/gliner-guard-omni) | `763f8aa9a7714dd2c3264f3751acdd5d854c5ac4` | GLiNER2 extractor. В корневом конфиге нет `auto_map.AutoModel`. |
| [GLiGuard 300M](https://huggingface.co/fastino/gliguard-LLMGuardrails-300M) | `7776816b54581f37b90dd2de12dbe13123229a9b` | GLiNER2 extractor. В корневом конфиге нет `auto_map.AutoModel`. |
| [OPIR Multitask Large](https://huggingface.co/knowledgator/opir-multitask-large-v1.0) | `69bb27407d66eab4797049d36ba75eef3579335b` | Собственный класс `GLiClassModel`. В корневом конфиге нет `auto_map.AutoModel`. |
| [OPIR Multitask Multilang](https://huggingface.co/knowledgator/opir-multitask-multilang-v1.0) | `1c1d66bd22e0ba93a435c710a9d9499be292cd65` | Собственный класс `GLiClassModel`. В корневом конфиге нет `auto_map.AutoModel`. |
| [Laya](https://huggingface.co/convaiinnovations/laya) | `fd4f9dbcf4fde222fd43174bdd4c3b08b789e4e1` | Собственный класс `LayaTypedDecisions`. В корневом конфиге нет `auto_map.AutoModel`. |

## Границы проверки

`plan → run → serve` проверен для HiveTraceGuard-Pro. Остальные успешные
прогоны выполнены напрямую через vLLM. Text-only режим HaloGuard охватывает
текстовую генерацию и не проверяет мультимодальные входы.

Для кастомных моделей [Transformers backend vLLM][custom] требует совместимую
реализацию и `auto_map.AutoModel`. Отсутствие точки входа в опубликованном
конфиге означает, что тест остановился на конфиге. Запуск через собственную
интеграцию и сохранение голов модели отдельно не проверялись. Поддержку
нативного класса сверяли с [реестром vLLM 0.34.0][registry].

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
