# Участие в проекте

## Сообщения коммитов

Используем [Conventional Commits](https://www.conventionalcommits.org/):
`<type>(<scope>): <description>`. Scope необязателен. Описание кратко
передаёт суть изменения.

- `feat` — новая возможность.
- `fix` — исправление ошибки.
- `docs` — документация.
- `refactor` — изменение структуры без изменения поведения.
- `test` — тесты.
- `build` — сборка и зависимости.
- `ci` — настройки CI.
- `chore` — остальные задачи сопровождения.

Примеры: `feat(cli): load datasets`, `fix(runtime): handle missing CUDA`.

Для несовместимого изменения добавьте `!` после типа или scope и опишите
изменение в теле коммита через `BREAKING CHANGE:`.

## Разработка и проверки

Для разработки по умолчанию используется Python 3.12. Выбирайте окружение
явно при каждом запуске. Для CPU:

```bash
uv run --extra inference --extra inference-cpu guardrail-tuner --help
uv run --extra inference --extra inference-cpu ruff check src tests/test_u1_cli_integration.py tests/test_export_execution.py tools/u1_cli_worker.py u1_benchmark/cli_integration.py u1_benchmark/team_rules.py
uv run --extra inference --extra inference-cpu ruff format --check src tests/test_u1_cli_integration.py tests/test_export_execution.py tools/u1_cli_worker.py u1_benchmark/cli_integration.py u1_benchmark/team_rules.py
uv run --extra inference --extra inference-cpu mypy
PYTHONPATH=src:.:tests uv run --extra inference --extra inference-cpu pytest tests/test_{application_workload,benchmark,cli,datasets,generation,manifest,new_modes,optimize,quality,rules,workload,u1_cli_integration,export_execution}.py
```

Для CUDA замените `inference-cpu` на `inference-cuda` во всех командах.
Варианты ONNX Runtime взаимоисключающие.

Полный набор У1 использует отдельное CPU-окружение с Transformers 5.20.1,
GLiNER2 2.1.0 и Laya 0.3.20. Рецепт и команда находятся в job `u1-cpu`
[CI](.github/workflows/ci.yml). Не устанавливайте их поверх CLI Transformers 4.x
или GPU lock. Новые интеграционные тесты выполняются в обоих стеках.
Проверка закреплённых gated-шаблонов требует заранее заполненного HF-кеша;
без него тест явно пропускается, это не GPU-приёмка. Остальные CPU-фикстуры
не скачивают веса и не запускают платные ресурсы.

Исторические исходники У1 добавлены без массового форматирования. Проверка
ruff охватывает ядро CLI и новые integration-модули; старые результаты и approvals
не перезаписываются. Новые У1-команды запускаются из checkout, а wheel сохраняет
старые самостоятельные команды.
