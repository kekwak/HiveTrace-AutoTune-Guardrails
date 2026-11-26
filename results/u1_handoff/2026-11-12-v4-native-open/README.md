# OpenGuardrails: native baseline на RTX 5090

12 ноября 2026. Новый baseline: BF16, HF-loader без принудительного attention
или SDPA math. Фактически во всех четырёх preflight trace:
`aten::_scaled_dot_product_efficient_attention`, CUTLASS memory-efficient kernels.
Веса/tokenizer: закреплённое зеркало 0124, template: прежний task2510.
Полные идентичности, версии и SHA256 архивов: [validation.json](validation.json);
исходники: [source_manifest.json](source_manifest.json).

| Бакет / batch | Статус | Наблюдаемый пик процесса, MiB | Worker с подготовкой, с |
| --- | --- | --- | --- |
| B1 / 64 | preflight_passed | 17612 | 100,17 |
| B2 / 32 | preflight_passed | 17626 | 82,04 |
| B3 / 16 | preflight_passed | 17658 | 89,25 |
| B4 / 8 | preflight_passed | 19072 | 88,39 |

Для каждого бакета сверены индексы 400 входов; inference выполнен на short/long/mixed
батчах, повторе после другого входа и точной границе 512/1024/2048/4096.
Проверены реальные input IDs/маски, native batching, strict parser, precision,
backend и отклонение превышения до forward. Это не полный inference всех 400
записей и не скоростная серия. Наблюдаемый пик с опросом 100 мс может пропустить
короткие всплески; Torch allocated/reserved сохранены отдельно.

Четыре ZIP скачаны; CRC, SHA256 и исходники проверены, offline-отчёты совпали
с сохранёнными. Архивы/полные trace доступны у У1 по приведённым SHA256, в Git
включены только компактные свидетельства. Пресеты позже разрежены до двух точек
на бакет; записанные smoke-конфиги и исходные архивы не менялись.

Это подтверждение исполнения верхних батчей **Open**, не измерение ускорения,
максимального batch или safety-качества. Другие новые baseline, compile/ORT/FLA
и overhead мониторинга этим запуском не проверены. Прежний FP16+math
[результат с OOM](../2026-11-12-v4-rtx5090/README.md) сохранён отдельно.
Pod остановлен после сохранения архивов; сетевой диск и среда сохранены.
