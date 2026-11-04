# HiveTrace-AutoTune-Guardrails
Найти ускорения guardrail-моделей без скрытого падения качества на редких атаках, обфускации и неанглийских текстах в продукте

## vLLM: адаптеры восьми моделей на T4

Новый [отчёт по адаптерам всех восьми моделей](docs/vllm-adapters.md):
экспериментальные decoder/pooling/custom-head интеграции и статус проверки.
На новой выборке 48 записей завершены T4-проверки KoalaAI, Prompt-Guard,
GLiNER, Laya и HiveTraceGuard: 34, 40, 28, 39 и 32 полных пары соответственно,
без смен решений. ShieldGemma с patch прошла отдельную выборку 12/12 по
четырём политикам, также без смен решений. Пропуски и ограничения ролей
явно учтены. OpenGuardrails остановился на дедлайне старта candidate,
HaloGuard — на ошибке native reference; восемь backend'ов не объявлены
проверенными или ускоренными.

Оставлены адаптеры, три GPU-скрипта сравнения и локальная сборка приватной
выборки (`src/vllm_probe/prepare_sample.py`). Скриптов выделения GPU и
управления Colab в новом наборе нет; команды воспроизведения — в отчёте.

## Laya через vLLM на T4

[Итог проверки](results/vllm-t4/LAYA-FINAL.md): экспериментальный FP32 decision-only
адаптер работает внутри vLLM. На 9711 примерах решения совпали с reference;
547 входов усечены. После выравнивания вычислений и группировки по длине
на двух физических T4 устойчивого преимущества по throughput не обнаружено.
Это исследовательский backend, не полная замена SDK/API; action-head не реализован.

## Профилирование Laya и ShieldGemma

- [Laya: ноутбук](notebooks/laya_profile.ipynb) · [статус и протокол](results/profiling/laya/README.md)
- [ShieldGemma-2B: ноутбук](notebooks/shieldgemma_profile.ipynb) · [статус и протокол](results/profiling/shieldgemma-2b/README.md)

Запуск в свежей Colab GPU session. `HF_TOKEN` добавить в Colab Secrets.
Обе модели выполнены на Colab T4; CSV и отчёты сохранены.
Laya: FP32-веса + SDK autocast FP16. ShieldGemma: FP32 + eager attention.
Разные задачи, инструкции и dtype: цифры не являются сравнением качества моделей.

Полные исходные baseline traces включены в `results/profiling/*/traces/baseline.zip`
с SHA256. [Пересчёт анализа и контрольные GPU-прогоны](src/profiling/README.md).
