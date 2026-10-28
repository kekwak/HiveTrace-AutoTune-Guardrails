# Проверка совместимости общей среды

Исходный запуск `team_env_smoke_45555111`: успешный smoke на Kaggle T4, не benchmark и не оценка качества. [Объяснение результатов](../../../docs/team_environment_smoke.md).

Опубликованы статус, параметры, install report, версии четырёх workers, численные результаты и подтверждения backend. Большие JSON сжаты gzip без изменения исходных байтов; после распаковки SHA256 сверяется с `source_sha256` в [индексе](artifact_index.json). Веса, экспорт ONNX и дубли трасс не включены. Подробные performance-traces предыдущей среды остаются в `results/comparison/` и `results/optimization/`.

`installation/install.json.gz` и четыре `stages/*/packages.json` — первичные источники полного lock. Успех этого запуска не подтверждает новое развёртывание из lock или работу нового skeleton на GPU. В `installation/pip_check.log` сохранено известное предупреждение sitecustomize/wrapt.
