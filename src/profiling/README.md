# Воспроизведение и контроль профилирования

Baseline notebooks: `notebooks/laya_profile.ipynb`, `notebooks/shieldgemma_profile.ipynb`.
Первый baseline сохранён без подмены контрольными повторами.

## Исходные доказательства — без GPU

Полные ZIP baseline (включая 20 исходных Chrome traces) находятся в git:

- `results/profiling/laya/traces/baseline.zip`;
- `results/profiling/shieldgemma-2b/traces/baseline.zip`.

Архивы по 18–20 MiB, каждый с `SHA256SUMS` рядом. После clone:

```bash
shasum -a 256 results/profiling/laya/traces/baseline.zip
shasum -a 256 results/profiling/shieldgemma-2b/traces/baseline.zip
python3 src/profiling/analyze.py results/profiling/laya/traces/baseline.zip results/profiling/laya
python3 src/profiling/analyze.py results/profiling/shieldgemma-2b/traces/baseline.zip results/profiling/shieldgemma-2b
```

`analyze.py` использует только Python stdlib. Проверяет CRC, p50/p95 по raw samples,
пересчитывает corrected kernel categories и атрибутирует GPU duration CPU operators
через `External id`. Результаты: `trace_audit.json` (включая SHA256 всех членов
архива), `operator_kernel_breakdown.csv`, `kernel_breakdown.csv`, `top_kernels_*`,
`figures/timeline_L16.svg`, `figures/timeline_L512.svg`. Kernel union/window/busy
пересчитываются в `gpu_profile_recomputed.csv` и сверяются с исходными числами.

SVG — аннотированные изображения из исходных trace events, **не скриншоты интерфейса
Perfetto**. Для исследования деталей извлечь `trace_L*.json` / `viz_*.json` и открыть
в https://ui.perfetto.dev. Корректно различать CPU enqueue spans и CUDA activity:
CPU D2H span включает ожидание ранее поставленных GPU операций.

## Независимые контрольные повторы — Colab GPU

Используется та же закреплённая модель и setup-ячейки baseline notebook.
`validate.py` выполняет только ячейки до загрузки модели, не весь baseline sweep.
Запускать в свежем kernel для каждого `VALIDATION_RUN=0,1,2` и каждой модели.
Число повторов относится к независимым kernels на одной VM, не к независимым GPU.

Пример с официальным `colab` CLI и существующей авторизацией:

```bash
colab new -s profiling-validation --gpu T4
colab upload -s profiling-validation notebooks/laya_profile.ipynb /content/profile.ipynb
colab upload -s profiling-validation src/profiling/validate.py /content/validate.py
colab exec -s profiling-validation --env VALIDATION_RUN=0 --timeout 1800 <<'PY'
import runpy
runpy.run_path('/content/validate.py', run_name='__main__')
PY
# Скачать ZIP по пути VALIDATION_ARCHIVE из вывода.
# Для run=1,2: restart-kernel, снова передать HF_TOKEN и установить VALIDATION_RUN.
```

HF_TOKEN должен быть в remote environment либо Colab Secrets; не вставлять его
в notebook source или shell history. Для ShieldGemma загружать соответствующий notebook.
После скачивания всех артефактов остановить только созданную для работы session.

Проверки:
- 3 независимых kernels, randomized порядок четырёх cases × forward/e2e;
- безопасный и опасный тексты одинаковых фактических длин 16/512;
- общий прогрев 40 запросов, 20 warmup + 50 samples на case/mode;
- GPU температура, clocks, power, utilization и memory раз в секунду;
- idle/peak allocated, incremental peak и reserved memory;
- Laya: FP32 reference, SDK probability agreement; `inference_mode`/`no_grad`,
  persistent autocast, FP16 weights ± autocast; отдельные targeted traces с shapes;
- между precision candidates восстанавливаются **все buffers**, включая
  nonpersistent RoPE `inv_freq`: одного `load_state_dict` после `.half()` недостаточно;
- Laya: 477/478/479 и опасный suffix за границей retention budget;
- ShieldGemma: last-token/full logits agreement, over-limit encode guard;
- настоящий SDK/scoring e2e trace и отдельные CUDA Events forward interval.

Кандидат, упавший на dtype/finite check, записывается в `ab_failures.json`,
не считается успешной оптимизацией. Latency A/B делается в run 0, поэтому
ускорение кандидата ещё нельзя считать подтверждённым между сессиями.
Неразмеченные sanity cases проверяют численные изменения, не качество модерации.

Скачанные follow-up ZIP назвать `laya-run-0.zip` … `laya-run-2.zip` и
`shieldgemma-2b-run-0.zip` … `shieldgemma-2b-run-2.zip`. Импорт и проверка:

```bash
python3 src/profiling/summarize_validation.py DOWNLOAD_DIRECTORY results/profiling
```

Скрипт сохраняет полные ZIP с SHA256, проверяет все 2400 repeat samples,
воспроизводит p50/p95 и показывает min/median/max **медиан независимых runs**.
Это не доверительный интервал production p95; pooling samples разных runs
не подменяет оценку межсессионной вариативности.

### Provenance исходника

Опубликованные контрольные прогоны **не содержат снимок или hash исполнявшегося
validate.py**. Это ограничение зафиксировано в `validation/provenance.json`.
Текущий скрипт не объявляется побайтно тождественным старому: в частности,
опубликованный ShieldGemma sanity run имеет 9 cases, текущий рецепт — 11.
Архивные цифры и README относятся к 9 cases.

Для будущих запусков команда выше использует `runpy` над реальным remote файлом:
сохраняются `executed_validate.py` и его SHA256 в env.json. При прямом CLI exec
без `__file__` snapshot не подделывается, отсутствие явно отмечается.

Разбор shapes и kernel time targeted Laya A/B:

```bash
python3 src/profiling/analyze_ab.py results/profiling/laya/validation/run-0/artifacts.zip results/profiling/laya/validation
```

`ab_copy_shapes.csv` связывает GPU copy kernels с `aten::copy_` и его input dims
через External id. `ab_profile_summary.csv` содержит counts и времена трёх
профилированных forward; их нельзя вычитать из обычного latency p50.
