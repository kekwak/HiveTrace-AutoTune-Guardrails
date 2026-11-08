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
uv run --extra inference --extra inference-cpu ruff check .
uv run --extra inference --extra inference-cpu ruff format --check .
uv run --extra inference --extra inference-cpu mypy
uv run --extra inference --extra inference-cpu pytest
```

Для CUDA замените `inference-cpu` на `inference-cuda` во всех командах.
Варианты ONNX Runtime взаимоисключающие.
