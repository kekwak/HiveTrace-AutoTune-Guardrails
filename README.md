# HiveTrace-AutoTune-Guardrails
Найти ускорения guardrail-моделей без скрытого падения качества на редких атаках, обфускации и неанглийских текстах в продукте

## Профилирование meta-llama/Llama-Prompt-Guard-2-22M

- [Результаты и узкие места](results/profiling/Llama-Prompt-Guard-2-22M/README.md)
- [Среда](env/versions.txt) — снимается скриптом [`src/env/snapshot.py`](src/env/snapshot.py)
- [`src/profiling/profile.py`](src/profiling/profile.py) — замеры задержки и трейсы PyTorch Profiler
- [`src/profiling/analyze_traces.py`](src/profiling/analyze_traces.py) — разбивка времени GPU по типам ядер
