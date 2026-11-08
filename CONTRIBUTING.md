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

## Проверки

Перед отправкой изменений выполните:

```bash
uv sync --extra hub --extra datasets --extra inference --extra inference-cpu
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv run --no-sync pytest
```
