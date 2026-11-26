# Замер качества внутри второй стадии CLI

Модуль `guardrail_tuner.quality` принимает те же варианты конфигурации, которые
вызывающая сторона использует для скорости, в виде двух батчевых предикторов.

```python
from guardrail_tuner.quality import Prediction, QualityRow, evaluate_quality


def adapt(predict, label_map):
    def predict_batch(batch):
        outputs = predict([item.text for item in batch])
        predictions = []
        for item, output in zip(batch, outputs, strict=True):
            verdict = label_map.get(output.get("label"))
            status = "ok" if verdict is not None else "invalid_output"
            predictions.append(Prediction(item.id, status, verdict, truncated=None))
        return predictions

    return predict_batch


rows = [
    QualityRow(
        row["id"],
        row["text"],
        {"harm": "unsafe", "safe": "safe"}[row["label"]],
        {key: row[key] for key in ("technique", "lang", "benchmark")},
    )
    for row in dataset
]
report = evaluate_quality(
    rows,
    adapt(baseline.predict, baseline_label_map),
    adapt(candidate.predict, candidate_label_map),
    batch_size=16,
    output_dir=run_dir / "quality",
    dataset_metadata={"source": source, "revision": resolved_revision},
)
```

Пример использует классификатор. Для генеративной модели вызывающая сторона
применяет её шаблон и преобразует текст ответа по контракту модели. Предиктор
возвращает один ответ на каждый входной ID. Допустимые статусы включают `ok`,
`over_limit`, `invalid_output` и `inference_error`. Передавайте `truncated=True`
при усечении, `False` при подтверждённом полном входе и `None`, если статус
неизвестен. Текущий позиционный предиктор CLI не возвращает эту информацию,
поэтому пример учитывает такие строки в `n_truncation_unknown`. Передавайте
`unsafe_score`, если доступна численная оценка направления unsafe.
Загрузку и освобождение моделей можно выполнять внутри предикторов лениво.

Каталог результата должен быть новым. В нём появятся два JSONL с предсказаниями,
`status.json`, `dataset.json` с ID, метками и срезами и `quality_report.json`.
Сводку можно вложить в отчёт второй стадии. Для пересчёта вызовите
`compare_predictions` со строками `QualityRow(text="", **row)` из поля `rows`
в `dataset.json`, его полем `metadata` и потоками `Prediction(**json.loads(line))`
из сохранённых JSONL. Для восстановления полного отчёта добавьте
`batch_size` из `status.json`.

Все доли указаны в диапазоне от 0 до 1, а дельты считаются как кандидат минус
эталон на общем покрытии. Отчёт даёт точные двусторонние тесты McNemar отдельно
для FNR и FPR. Holm применяется отдельно для каждой метрики и оси срезов.
MDE является оценкой по нормальному приближению при alpha 0,05, мощности 0,8
и доле улучшившихся вердиктов 0,005. Критерии приёмки применяет вызывающая сторона.
