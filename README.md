# HiveTrace-AutoTune-Guardrails
Найти ускорения guardrail-моделей без скрытого падения качества на редких атаках, обфускации и неанглийских текстах в продукте

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
