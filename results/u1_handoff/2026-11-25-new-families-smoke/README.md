# ProtectAI и AccuKnox: smoke на RTX 5090

25 ноября 2026, RTX 5090 32607 MiB, драйвер 595.112.07, GPU 0.
Только native baseline и короткая диагностика. Benchmark-серии, кандидаты
compile/ORT/FLA и quality не запускались. Batch 2 выбран заранее для smoke;
это не проверка верхних размеров основной сетки.

| Проверка | Результат | Доказательство |
| --- | --- | --- |
| Установка `team-d3` из lock | passed: inventory, pip check, CUDA primitive | [d3-env.zip](d3-env.zip) |
| ProtectAI, B1 x 2, `team-rtx` | worker failed: AcceleratorError; отчёт incomplete | [protectai-results.zip](protectai-results.zip) |
| AccuKnox, B1-B4 x 2, `team-d3` | 4 workers failed: AcceleratorError; отчёт incomplete | [accuknox-results.zip](accuknox-results.zip) |

## Диагностика

[diagnostics.zip](diagnostics.zip) содержит отдельные diagnostic-only результаты,
скрипты и логи. Они не заменяют preflight и не получают approval.

- ProtectAI через адаптер воспроизводимо падает с CUDA illegal memory access.
  Отдельные простые FP32 linear-операции проходят. Прямой вызов через Transformers
  на коротком тексте проходит как с синхронной отладкой, так и без неё; последний
  контроль сохраняет `CUBLAS_WORKSPACE_CONFIG=:4096:8` и `OMP_NUM_THREADS=2`.
  Это сужает поиск до различий загрузки/вызова, но не устанавливает конкретную
  причину. Асинхронные traceback указывают на разные операции и сами по себе
  не доказывают неисправность attention или cuBLAS.
- AccuKnox с `CUDA_LAUNCH_BLOCKING=1` падает до первого запроса:
  `D3.__init__ -> from_pretrained -> _materialize_copy -> tensor.to(device, dtype)`.
  Затем та же CUDA-ошибка возникает при восстановлении RNG. Причина незаконного
  доступа при загрузке ещё не установлена; это не зарегистрированный OOM.
- Первая отдельная диагностика AccuKnox остановилась на `OfflineModeIsEnabled`:
  проверка весов запрашивает HF metadata. Попытка сохранена; последующая диагностика
  с доступом к HF воспроизвела исходную CUDA-ошибку. Это не смена runtime или lock.

Настройки моделей и зависимости для получения успешного результата не менялись.
Установка d3 подтверждена; успешная работа обеих моделей через измеритель на этой
среде **не подтверждена**. Нужен разбор загрузки и вызова до нового smoke.

## Целостность и ресурсы

[validation.json](validation.json): CRC/SHA256 четырёх архивов, совпадение 83
исходников каждого smoke-архива с проверенным локальным снимком и полное совпадение
обоих offline-пересобранных отчётов. Архивы без весов, venv и credentials.

Pod `c4p7lqu1cm90av` остановлен; `EXITED` подтверждён через Runpod после остановки
в 14:42 UTC. Pod и сетевой диск сохранены. Среда и HF-кеш остаются на диске.
Публикация и push не выполнялись.
