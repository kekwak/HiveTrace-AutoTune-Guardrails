# GLiNER: eager / compile / ONNX

Статус: completed. Одна T4-сессия; не окончательное подтверждение.
FP32, batch 1. Численная регрессия на development; не safety-оценка. Holdout не использован.
p50/p95 — медианы трёх блочных квантилей. Отношения рассчитаны внутри пар.

| Пара | Вход | p50 контроль → кандидат, мс | p95 контроль → кандидат, мс | p50 кандидат/контроль | Статус |
| --- | --- | --- | --- | --- | --- |
| onnx_vs_compile | legacy_short | 18.09 → 9.86 | 19.16 → 10.67 | 0.542 | promising_requires_confirmation |
| onnx_vs_compile | varying_short | 18.08 → 9.83 | 19.23 → 10.59 | 0.539 | promising_requires_confirmation |
| onnx_vs_compile | legacy_medium | 19.38 → 15.67 | 20.13 → 16.42 | 0.808 | promising_requires_confirmation |
| onnx_vs_compile | varying_medium | 19.32 → 16.41 | 20.51 → 17.56 | 0.844 | promising_requires_confirmation |
| onnx_vs_compile | legacy_long | 43.05 → 45.07 | 46.18 → 46.82 | 1.047 | inconclusive |
| onnx_vs_compile | varying_long | 46.05 → 49.80 | 48.89 → 51.99 | 1.082 | inconclusive |
| onnx_vs_compile | mixed | 21.73 → 17.78 | 46.54 → 50.33 | 0.818 | inconclusive |
| onnx_vs_eager | legacy_short | 47.96 → 9.87 | 50.10 → 11.01 | 0.207 | promising_requires_confirmation |
| onnx_vs_eager | varying_short | 48.33 → 10.08 | 53.11 → 11.24 | 0.214 | promising_requires_confirmation |
| onnx_vs_eager | legacy_medium | 48.36 → 15.69 | 51.11 → 16.29 | 0.324 | promising_requires_confirmation |
| onnx_vs_eager | varying_medium | 48.21 → 16.27 | 50.97 → 17.12 | 0.337 | promising_requires_confirmation |
| onnx_vs_eager | legacy_long | 50.64 → 44.83 | 53.24 → 46.61 | 0.880 | promising_requires_confirmation |
| onnx_vs_eager | varying_long | 54.33 → 49.03 | 57.44 → 50.76 | 0.902 | inconclusive |
| onnx_vs_eager | mixed | 48.70 → 17.26 | 55.44 → 49.63 | 0.353 | promising_requires_confirmation |
| compile_vs_eager | legacy_short | 47.74 → 18.16 | 50.61 → 21.05 | 0.380 | promising_requires_confirmation |
| compile_vs_eager | varying_short | 48.01 → 18.10 | 51.67 → 20.05 | 0.377 | promising_requires_confirmation |
| compile_vs_eager | legacy_medium | 48.40 → 19.40 | 51.24 → 20.43 | 0.399 | promising_requires_confirmation |
| compile_vs_eager | varying_medium | 49.14 → 19.39 | 51.56 → 20.25 | 0.394 | promising_requires_confirmation |
| compile_vs_eager | legacy_long | 50.78 → 43.71 | 53.28 → 46.82 | 0.861 | promising_requires_confirmation |
| compile_vs_eager | varying_long | 54.48 → 45.55 | 57.68 → 48.09 | 0.836 | promising_requires_confirmation |
| compile_vs_eager | mixed | 49.03 → 21.83 | 55.22 → 45.96 | 0.447 | promising_requires_confirmation |

Для выбора между ONNX и compile смотреть только onnx_vs_compile; не делить результаты двух отдельных сравнений с eager. Inconclusive не означает равенство.
Память PyTorch не включает ORT; telemetry и подготовка находятся внутри jobs. Экспорт/компиляция исключены из latency; сравнение не доказывает окупаемость холодного старта.
