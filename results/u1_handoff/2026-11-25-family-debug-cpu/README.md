# Разбор ошибок новых семейств

Это продолжение [неуспешного GPU-smoke](../2026-11-25-new-families-smoke/README.md),
а не исправленная GPU-приёмка. Production-код, зависимости и исходные архивы не менялись.

## Проверено на CPU

[ProtectAI parity](protectai_cpu_parity.r2.json), настоящий закреплённый checkpoint:
все 202 тензора весов совпали по SHA256; token IDs, маски и logits двух путей
совпали на трёх нейтральных входах, включая пустой. Максимальная разница logits 0.
Проверены загрузка через штатные HF-параметры и `hf_load_options`, затем native
и adapter preparation/forward. CPU-устройство задаётся только диагностике;
production-адаптер не получил CPU fallback.

Среда: macOS, Torch 2.15.0, Transformers 5.20.1, Tokenizers 0.23.3.
Это не целевая CUDA-среда; результат не исключает GPU-специфическую проблему.
[Скрипт](protectai_cpu_parity.py) запускается с корнем репозитория в `PYTHONPATH`
и доступными в HF-кеше весами. Первичная попытка сохранения не поддержала set
в `loading_info`; JSON r2 исправляет только эту сериализацию, не вычисления.

## Следующая CUDA-проверка

[diagnose_gpu.py](diagnose_gpu.py) выполняет ровно один явно выбранный случай
в свежем процессе, сохраняет фазу ошибки, traceback и среду. Синхронизация между
фазами и `CUDA_LAUNCH_BLOCKING=1` используются только для диагностики.
Флаг `--loading serial` явно проверяет штатный `HF_DEACTIVATE_ASYNC_LOAD=1`;
это гипотеза для AccuKnox, не применённое исправление или автоматический fallback.

Для ProtectAI маршруты последовательно отделяют repo/local source, старый dtype
аргумент, loading-info, подготовку входов и полный адаптер. Для AccuKnox доступны
native SDK и адаптер, с обычной либо последовательной загрузкой. Никакой
диагностический результат не получает approval. Запускать только выбранные случаи,
не всю матрицу автоматически; веса, dtype и backend не заменять.

Пример для существующего подготовленного Pod, после отдельного запуска GPU:

```bash
ROOT=/workspace/u1-new-families-20261125
PYTHONPATH="$ROOT/repo" HF_HOME="$ROOT/hf-cache" timeout 180 \
  "$ROOT/d3-env/venv/bin/python" diagnose_gpu.py \
  --repo "$ROOT/repo" --config "$ROOT/accuknox-smoke.json" \
  --environment "$ROOT/d3-env" --route adapter --loading serial \
  --output "$ROOT/debug-accuknox-serial"
```

Reproducer подготовлен локально; он ещё не загружен на Pod и не выполнялся на GPU.
Две попытки старта Pod получили отказ по доступности. Очередь не создавалась,
`EXITED` подтверждён повторным чтением. [Контрольные суммы](validation.json).
