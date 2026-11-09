# guard_eval — качество, смены вердиктов и скорость guardrail-моделей с динамическим квантованием

Прогоняет любую guardrail-модель (и её квантованные версии) на датасете хвостов `AutotuneGuardrails/tails`,
считает метрики по сплитам, бенчмаркам и в целом, сравнивает две конфигурации по числу изменённых вердиктов
и измеряет скорость по бакетам длины.

## Запуск на Kaggle (2× T4)

Ноутбук [`kaggle/guard_eval_kaggle.ipynb`](kaggle/guard_eval_kaggle.ipynb) ничего не отправляет в git: все результаты
собираются в архив, который скачивается вручную.

1. **Код.** Загрузить `guard-eval-code.zip` (архив этой ветки: `git archive --format=zip -o guard-eval-code.zip HEAD`)
   как приватный Kaggle Dataset (Datasets → New Dataset → файл, видимость Private; имя любое — ноутбук ищет код по всему
   `/kaggle/input`). После изменений кода — загрузить новую версию датасета.
   Если ветка есть на GitHub, этот шаг можно пропустить — ноутбук её склонирует.
2. Импортировать ноутбук: Create → Import Notebook → `guard_eval_kaggle.ipynb`. Add Input → этот датасет.
3. Settings: Accelerator **GPU T4 x2**, Internet **On**.
4. Add-ons → Secrets: `HF_TOKEN` — HF-токен с доступом к `AutotuneGuardrails/tails`, привязать к ноутбуку.
5. В ячейке настроек при необходимости поменять `MODELS`, `METHODS`, `EXTRA_ARGS`. Для первой проверки всей цепочки:
   `EXTRA_ARGS = "--limit-per-split 20"` (минуты вместо часов).
6. **Save Version → Save & Run All (Commit)** — прогон идёт в фоне, браузер можно закрыть (лимит сессии 12 ч).
   Интерактивно (Run All в редакторе или из Cursor через Kaggle Jupyter Server) тоже работает, если сессия не закроется.
7. Скачать из Output `guard_eval_results.zip`: `results/quality/runs/` (прогоны), `reports/` (`SUMMARY.md`, отчёты
   сравнений — есть тексты из закрытых наборов, не публиковать), `logs/`.

Если сессия оборвалась в интерактивном режиме — запустить снова в той же сессии: готовые тексты берутся из кэша.
Квота Kaggle — 30 ч GPU в неделю на аккаунт; модели можно разделить между участниками через `MODELS` — папки прогонов
у разных моделей и методов не пересекаются, архивы просто складываются.

**Особенности T4 (sm_75):** точность fp16 (`dtype: auto`); `torchao_fp8_dyn` не поддерживается (нужен sm_89+) и помечается
без загрузки модели; поддержка INT8-ядер torchao и bitsandbytes на T4 проверяется `python -m guard_eval check`;
WildGuard не входит в набор по умолчанию — эталон 7B в fp16 (14.5 ГБ) не помещается в 15 ГБ.
Скорость на T4 — предварительная: продукт работает на Blackwell, итоговые замеры — на RTX 5090.

## Установка вручную (RunPod или локально)

```bash
git clone https://github.com/kekwak/HiveTrace-AutoTune-Guardrails.git && cd HiveTrace-AutoTune-Guardrails
git switch quality/guard-eval
pip install -r requirements.txt
export HF_TOKEN=...            # нужен доступ к AutotuneGuardrails/tails
python -m guard_eval check     # GPU, версии и проверка всех методов квантования на маленькой сети
python -m pytest -q tests      # логика метрик, кэша и spec (CPU)
```

Веса моделей кэшируются в `~/.cache/huggingface`; на RunPod укажите `HF_HOME` на network volume, чтобы не качать WildGuard (14.5 ГБ) заново.

## Модели и методы

| Псевдоним | Модель | Адаптер |
|---|---|---|
| `gliner-guard` | hivetrace/gliner-guard-omni | `gliner` — logits классификатора safety |
| `koala` | KoalaAI/Text-Moderation | `seq_cls` — P(unsafe) = 1 − P(OK) |
| `hivetrace-guard-pro` | hivetrace/HiveTraceGuard-Pro | `causal_verdict` — логиты `safe`/`unsafe` первого токена |
| `openguardrails` | openguardrails/OpenGuardrails-Text-4B-0124 | `causal_verdict` — шаблон версии 2510 (инструкция классификатора), `safe`/`unsafe` |
| `wildguard` | allenai/wildguard | `causal_verdict` — логиты ` yes`/` no` после `Harmful request:`; только на GPU ≥ 24 ГБ |

Методы (`--quant`): `none` (эталон), `torchao_int8_dyn` (W8A8, активации int8 на лету), `torchao_fp8_dyn`
(FP8 e4m3 по строкам, нужен sm_89+), `bnb_int8` (LLM.int8(), выбросы в fp16). Квантуются `nn.Linear`;
классификационные головы и `lm_head` остаются в исходной точности.

Точность — `dtype: auto`: bf16 на GPU с sm_80+, fp16 на T4, fp32 на CPU. Явный `--set dtype=bfloat16` на T4 даёт ошибку,
а не медленную эмуляцию. `--set compile=true` включает `torch.compile` основного модуля (у GLiNER — encoder): без
компиляции torchao квантует активации отдельными ядрами и почти не ускоряет модель.

## Команды

**Квантизация любой модели любым методом** (прогон качества квантованной версии + отчёт):
```bash
python -m guard_eval run --model wildguard --quant torchao_fp8_dyn
python -m guard_eval run --model koala --quant bnb_int8 --speed          # плюс замер скорости
```

**Прогон любой модели** (без квантования; можно менять любое поле spec):
```bash
python -m guard_eval run --model hivetrace-guard-pro
python -m guard_eval run --model koala --set dtype=float32 --set batch_size=1
python -m guard_eval run --model specs/models/my_model.yaml               # своя модель
python -m guard_eval run --model koala --limit-per-split 50 --benchmarks rtplx csrt   # быстро, на подвыборке
```

**Скорость** (бакеты synthetic-v1, как у У1: legacy/varying short·medium·long и mixed):
```bash
python -m guard_eval speed --model koala --quant torchao_int8_dyn
```

**Сравнение двух конфигураций** — недостающие прогоны досчитываются, готовые берутся из кэша:
```bash
python -m guard_eval compare --model koala --a none --b torchao_int8_dyn --speed       # эталон против квантования
python -m guard_eval compare --model koala --set-a dtype=float32 --set-b dtype=bfloat16  # одна модель, разные настройки
python -m guard_eval compare --spec-a koala --spec-b gliner-guard                        # две разные модели
```

**Одна модель всеми тремя методами** (качество + скорость + сравнения с эталоном):
```bash
./scripts/quantize_model.sh koala
METHODS="bnb_int8 torchao_int8_dyn" ./scripts/quantize_model.sh wildguard --limit-per-split 50
```

**Все модели всеми методами + замеры скорости + общая сводка:**
```bash
./scripts/run_all.sh                                   # одна GPU; логи в logs/, сводка в reports/SUMMARY.md
./scripts/run_all_2gpu.sh                              # две GPU: качество параллельно, скорость по очереди
MODELS="koala gliner-guard" ./scripts/run_all.sh
SPEED=0 ./scripts/run_all.sh                           # только качество
python -m guard_eval summary                           # пересобрать сводку из готовых отчётов
```
По умолчанию `MODELS="gliner-guard koala hivetrace-guard-pro openguardrails"`; на 5090 добавьте `wildguard`.

## Что сохраняется

- `results/quality/runs/<модель>/<spec_hash>/` — **коммитится в git**: `predictions.parquet` (только `text_hash` и
  скоры, без текстов и меток), `manifest.json` (spec, среда, GPU, версии, сведения о квантовании), `speed.json`.
  `spec_hash` включает всё, что меняет скоры: модель, revision, dtype, квантование, шаблон, батч, max_length,
  версию кода, класс GPU и версии библиотек. Порог в ключ не входит — он применяется при подсчёте метрик.
  Повторный запуск считает только недостающие тексты; строки с ошибкой пересчитываются.
- `reports/` — **только локально** (в `.gitignore`): `summary.md` и CSV по сплитам, бенчмаркам, хвостам, типам
  атак и языкам, `flips.csv` с текстами изменённых вердиктов, `SUMMARY.md` по всем сравнениям.

Прогоны из скачанного архива можно вручную положить в `results/quality/runs/` своей копии репозитория: повторные
запуски возьмут их из кэша, а отчёты и сравнения можно пересобрать локально (`compare`, `summary`).

## Метрики

Вердикт: unsafe, если P(unsafe) > порог (по умолчанию 0.5; при равенстве — safe). Строки с ошибкой исключаются и
считаются отдельно. Нулевой знаменатель даёт «—» (не определено), а не 0.

- **Уровни:** сплит, бенчмарк, хвост (`tail`), тип атаки, язык; итог — pooled (все примеры вместе) и macro
  (среднее по бенчмаркам, каждый с одинаковым весом).
- **Одна модель:** confusion, FNR, FPR, precision, F1 по классам, macro-F1, balanced accuracy, ROC-AUC.
- **Сравнение:** число и доля изменённых вердиктов (safe→harm, harm→safe; хуже — был верный, стал неверный;
  лучше — наоборот), тест McNemar, ΔFNR / ΔFPR / Δmacro-F1, |Δp|, число примеров у порога.
- **Интервалы:** 95% bootstrap (1000 повторов, seed 42); для сравнения — парный bootstrap по тем же примерам,
  для macro — с ресэмплингом внутри каждого бенчмарка.

## Своя модель

Скопируйте YAML из `specs/models/` и укажите `adapter`:
- `seq_cls` — `safe_labels` (классы, которые считаются безопасными);
- `causal_verdict` — `prompt` (`chat_template` или `format` с `{text}`) и `verdict` (`prefix`, токены `safe`/`unsafe`);
  id токенов вердикта определяются автоматически в контексте промпта;
- `gliner` — `task` и `labels`.

## Ограничения

- Скорость меряется по одной конфигурации на процесс, без парного A/B и A/A из протокола v1 У1. Отчёт сравнивает
  отдельные прогоны; для итоговых заявлений о ≥20% нужен парный протокол.
- Тексты длиннее `max_length` обрезаются (у decoder — только текст пользователя, не инструкция) и помечаются
  `truncated`.
- Проверено на CPU: все адаптеры (вердикты, совпадение батча и одиночных запросов), токены вердикта и промпты
  OpenGuardrails/WildGuard, метрики, кэш, сравнение, отчёты, bash-сценарии. **На GPU не запускалось:** методы
  квантования, `compile`, замеры скорости на CUDA и ноутбук Kaggle — начните с `python -m guard_eval check`.
