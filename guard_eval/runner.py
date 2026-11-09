"""Прогон качества с кэшем: считаются только тексты, которых ещё нет в predictions.parquet."""

import time
import traceback

from . import environment
from .adapters import make_adapter
from .cache import Run
from .quant import UnsupportedQuant, check_support
from .spec import run_label, spec_key


def prepare_run(spec, device):
    env = environment.describe(device)
    return Run(spec, spec_key(spec, env)), env


def run_quality(spec, dataset, device, save_every=512, log=print):
    run, env = prepare_run(spec, device)
    todo = run.missing(dataset["text_hash"])
    label = run_label(spec)
    if not todo:
        log(f"[{label}] все {dataset['text_hash'].nunique()} текстов уже посчитаны: {run.dir}")
        return run
    log(f"[{label}] нужно посчитать {len(todo)} текстов → {run.dir}")

    texts = dict(zip(dataset["text_hash"], dataset["text"]))
    # сортировка по длине уменьшает паддинг в батчах
    todo.sort(key=lambda h: len(texts[h]))
    run.update_manifest(spec=spec, spec_hash=run.key, environment=env, status="running")
    reason = check_support(spec.get("quant", "none"), device)
    if reason:   # не скачиваем и не грузим модель ради заведомо неподдерживаемого метода
        run.update_manifest(status="unsupported", error=reason)
        log(f"[{label}] метод не поддерживается: {reason}")
        return run
    adapter = make_adapter(spec, device)
    try:
        started = time.perf_counter()
        adapter.load()
        run.update_manifest(quant=adapter.quant_info, load_seconds=round(time.perf_counter() - started, 1))
    except UnsupportedQuant as e:
        run.update_manifest(status="unsupported", error=str(e))
        log(f"[{label}] метод не поддерживается: {e}")
        return run

    batch_size, buffer, done = spec["batch_size"], [], 0
    started = time.perf_counter()
    try:
        for i in range(0, len(todo), batch_size):
            hashes = todo[i:i + batch_size]
            try:
                results = adapter.predict([texts[h] for h in hashes])
                buffer += [{**r, "text_hash": h, "error": None} for h, r in zip(hashes, results)]
            except Exception as e:
                if batch_size == 1 or len(hashes) == 1:
                    buffer.append({"text_hash": hashes[0], "p_unsafe": None, "n_tokens": None, "truncated": None,
                                   "error": f"{type(e).__name__}: {e}"[:500]})
                else:   # повторяем батч поштучно, чтобы ошибка одного текста не теряла остальные
                    for h in hashes:
                        try:
                            buffer.append({**adapter.predict([texts[h]])[0], "text_hash": h, "error": None})
                        except Exception as one:
                            buffer.append({"text_hash": h, "p_unsafe": None, "n_tokens": None, "truncated": None,
                                           "error": f"{type(one).__name__}: {one}"[:500]})
            done += len(hashes)
            # систематическая ошибка: первые батчи целиком упали — дальше считать бессмысленно
            if done <= 2 * batch_size and done == len(buffer) and all(r["error"] for r in buffer):
                if done >= min(2 * batch_size, len(todo)):
                    error = buffer[0]["error"]
                    run.update_manifest(status="failed", error=error)
                    log(f"[{label}] ОСТАНОВЛЕНО: все первые {done} текстов упали с ошибкой: {error}")
                    return run
            if len(buffer) >= save_every or done == len(todo):
                run.append(buffer)
                buffer = []
                rate = done / (time.perf_counter() - started)
                log(f"[{label}] {done}/{len(todo)}  ({rate:.1f} текстов/с)")
        run.update_manifest(status="complete", predicted=len(run.predictions()),
                            errors=int(run.predictions()["error"].notna().sum()))
    except BaseException:
        if buffer:
            run.append(buffer)
        run.update_manifest(status="interrupted", traceback=traceback.format_exc()[-2000:])
        raise
    finally:
        adapter.close()
    return run
