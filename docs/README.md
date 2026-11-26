# Документация CLI и У1

Актуальная редакция: 26 ноября 2026. Конфиг schema 4, результаты 5,
speed `u1-static-batch-v4-reuse-v1`, quality 2. Номера версий корпуса и
измерителя независимы. Общий CLI подключён к У1 и влит в main.
Полные baseline ProtectAI/AccuKnox прошли на RTX 5090; ProtectAI завершил новый
eval с `reference_pass`. [Текущая GPU-проверка](../results/u1_handoff/2026-11-26-cli-eval/README.md)
отделена от исторических smoke и качества по разметке.

## Текущие документы

- [Команды за минуту](cli_quick_reference.md): назначение команд и основной маршрут.
- [Запуск измерителя](u1_handoff.md): единственная операционная инструкция.
- [Подробное руководство CLI](cli-guide.md): включая прежние самостоятельные режимы.
- [Протокол](measurement_protocol_v1.md): стадии, критерии, шум, память и quality.
- [Baseline-режимы](baseline_modes.md): актуален верхний раздел schema 4.
- [Состояние роли и следующие шаги](u1_protocol_plan.md): актуален верхний раздел.
- [Корпус](speed_dataset.md): модельные бакеты и отличие от старых пересечений.
- [Готовность](u1_handoff_readiness.md): датированные проверки; старый smoke не
  подтверждает новый lifecycle.

Для новых запусков используйте [configs/v4](../configs/v4) и примеры
[speed import](../examples/u1_v4_speed_import.json),
[quality import](../examples/u1_v4_quality_import.json),
[quality config](../examples/u1_v4_quality.json),
[external config](../examples/u1_v4_external.json) и
[external adapter](../examples/u1_v4_external_adapter.py).
Другие, более ранние JSON-примеры и configs не переводились автоматически на schema 4.

## Архив и карточки

- [Протокол v0.1](measurement_protocol.md), [screening 30–31 октября](benchmark.md),
  [T4 skeleton](team_environment.md) и [исходный smoke среды](team_environment_smoke.md)
  описывают свои этапы, не текущие команды/критерии.
- [Тезисы и текст выступления 4 ноября](u1_sync_brief.md) сохранены как архив;
  их нельзя переносить в новую презентацию как текущий статус без обновления.
- [HF-карточка v3](hf_dataset/README.md) и [карточка ролевого v4](hf_dataset/README_v4.md)
  описывают соответствующие снимки данных. Относительные ссылки `REPORT.md` и
  `contracts.json` в экспортной v3-карточке разрешаются в HF-выпуске, не в `docs/`.
  Текущие указания использования в локальной карточке v4 не означают публикацию на HF.
- Содержимое `results/`, `u1/` и исторических notebooks сохраняет свои числа,
  исходники и approvals; они не переоцениваются по новым критериям задним числом.

Основной маршрут: **baseline один раз → candidate с `--baseline-ref` → report/table**.
Confirm необязателен. Полный baseline v4 поддерживается с проверкой совместимости;
старые schema-3 конфиги и архивы текущий CLI не принимает.
