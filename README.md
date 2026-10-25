# HiveTrace-AutoTune-Guardrails
Найти ускорения guardrail-моделей без скрытого падения качества на редких атаках, обфускации и неанглийских текстах в продукте

## Профилирование Laya и ShieldGemma

- [Laya: ноутбук](notebooks/laya_profile.ipynb) · [статус и протокол](results/profiling/laya/README.md)
- [ShieldGemma-2B: ноутбук](notebooks/shieldgemma_profile.ipynb) · [статус и протокол](results/profiling/shieldgemma-2b/README.md)

Запуск в свежей Colab GPU session. `HF_TOKEN` добавить в Colab Secrets.
Обе модели выполнены на Colab T4; CSV и отчёты сохранены.
Laya: FP32-веса + SDK autocast FP16. ShieldGemma: FP32 + eager attention.
Разные задачи, инструкции и dtype: цифры не являются сравнением качества моделей.

Полные исходные baseline traces включены в `results/profiling/*/traces/baseline.zip`
с SHA256. [Пересчёт анализа и контрольные GPU-прогоны](src/profiling/README.md).
