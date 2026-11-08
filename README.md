# HiveTrace-AutoTune-Guardrails
Найти ускорения guardrail-моделей без скрытого падения качества на редких атаках, обфускации и неанглийских текстах в продукте

## Скелет CLI

Python 3.10 или новее. Для локального каталога модели достаточно базовой установки.
Дополнение `hub` позволяет читать метаданные Hugging Face.

```bash
python -m pip install -e '.[hub]'
guardrail-tuner inspect KoalaAI/Text-Moderation
guardrail-tuner plan KoalaAI/Text-Moderation
guardrail-tuner plan ./local-checkpoint
```

`inspect` читает `config.json`. Для Hugging Face можно указать `--revision`.
На этом этапе CLI загружает только конфиг.

`plan` предлагает проверки по типу архитектуры из конфига. Статус
`unverified` означает, что совместимость нужно подтвердить запуском.
`needs_inference_contract` означает, что CLI пока нужен способ подготовить
вход и сравнить выход модели с эталоном.

Следующий этап — выполнять эти проверки, а затем измерять прошедшие варианты
на заданной нагрузке. Результаты текущего `plan` описывают кандидатов.
