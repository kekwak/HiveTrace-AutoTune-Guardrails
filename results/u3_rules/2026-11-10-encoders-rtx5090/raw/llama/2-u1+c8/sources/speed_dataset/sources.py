"""Bounded, reproducible EN/RU pool; text selection precedes model tokenization."""

import hashlib
import random
import re
from pathlib import Path

from .core import CONTRACTS, SEED, digest, file_hash, implementation_hash, validate_pool, write_json, write_jsonl

WIKI_REPO = "wikimedia/wikipedia"
WIKI_REV = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
SHARDS = {"en": "20231101.en/train-00000-of-00041.parquet",
          "ru": "20231101.ru/train-00000-of-00021.parquet"}
LICENSES = ["CC-BY-SA-3.0", "GFDL"]


def text_record(text, language, source, provenance, form):
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {"id": h, "text_sha256": h, "text": text, "language": language,
            "source": source, "provenance": provenance, "format": form}


def excerpt(article, length, seed):
    """Contiguous word-aligned slice, with offsets into the unchanged article."""
    matches = list(re.finditer(r"\S+", article))
    if not matches or len(article) < length:
        return None
    rng = random.Random(seed)
    starts = [m.start() for m in matches if m.start() + length <= len(article)]
    if not starts:
        return None
    start = starts[rng.randrange(len(starts))]
    end = start + length
    while end < len(article) and not article[end].isspace():
        end += 1
    return article[start:end], start, end


def target_chars(language, ordinal, rng, long_only=False):
    if long_only:
        return rng.randint(*{"en": (9500, 14500), "ru": (6500, 10000)}[language])
    ranges = {"en": [(60, 260), (1250, 2600), (3100, 5900)],
              "ru": [(40, 160), (600, 1700), (1700, 3600)]}
    return rng.randint(*ranges[language][ordinal % 3])


def wiki_records(path, language, count, excluded=None, excluded_articles=None, long_only=False):
    import pyarrow.parquet as pq
    result, seen, article_ids = [], set(), set(excluded_articles or ())
    excluded = set() if excluded is None else excluded
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["id", "url", "title", "text"]):
        for article in batch.to_pylist():
            aid = str(article["id"])
            if aid in article_ids:
                continue
            ordinal = len(result)
            rng = random.Random(f"{SEED}:{language}:wiki:{ordinal}")
            chosen = excerpt(article["text"], target_chars(language, ordinal, rng, long_only), f"{SEED}:{aid}")
            if chosen is None:
                continue
            text, start, end = chosen
            normalized = " ".join(text.casefold().split())
            if normalized in seen or digest(text.encode()) in excluded:
                continue
            seen.add(normalized)
            article_ids.add(aid)  # one excerpt/article: no near-identical overlapping windows
            provenance = {"dataset": WIKI_REPO, "revision": WIKI_REV, "shard": SHARDS[language],
                          "article_id": aid, "title": article["title"], "url": article["url"],
                          "authors": "Wikipedia contributors; see article history",
                          "history_url": article["url"] + "?action=history", "licenses": LICENSES,
                          "start_char": start, "end_char": end, "article_sha256": digest(article["text"].encode()),
                          "transformation": "contiguous word-aligned excerpt; original spelling retained"}
            result.append(text_record(text, language, "wikipedia", provenance, "excerpt"))
            if len(result) == count:
                return result
    raise ValueError(f"Pinned first shard exhausted: {language}, {len(result)}/{count}")


VOCAB = {
    "en": {
        "places": ["library", "museum", "workshop", "garden", "archive", "laboratory", "station", "school", "observatory", "office", "gallery", "theatre", "warehouse", "cafe", "market", "clinic"],
        "objects": ["schedule", "catalogue", "report", "map", "invoice", "manual", "application", "diagram", "table", "reservation", "inventory", "checklist", "budget", "delivery", "feedback", "contract"],
        "actions": ["review", "compare", "describe", "summarize", "update", "translate", "organize", "check", "discuss", "prepare", "explain", "archive"],
        "conditions": ["before the meeting", "after the inspection", "with the coordinator", "using the revised dates", "without personal details", "for the new visitors", "during the morning shift", "in chronological order", "with source references", "after checking the units", "including the exceptions", "with a short explanation"],
    },
    "ru": {
        "places": ["библиотека", "музей", "мастерская", "сад", "архив", "лаборатория", "станция", "школа", "обсерватория", "офис", "галерея", "театр", "склад", "кафе", "рынок", "клиника"],
        "objects": ["расписание", "каталог", "отчёт", "карта", "счёт", "инструкция", "заявка", "схема", "таблица", "бронирование", "опись", "список", "бюджет", "доставка", "отзыв", "договор"],
        "actions": ["проверить", "сравнить", "описать", "сократить", "обновить", "перевести", "упорядочить", "обсудить", "подготовить", "объяснить", "сохранить", "согласовать"],
        "conditions": ["до встречи", "после осмотра", "вместе с координатором", "с новыми датами", "без персональных данных", "для новых посетителей", "в утреннюю смену", "в хронологическом порядке", "со ссылками на источники", "после проверки единиц", "с учётом исключений", "с кратким пояснением"],
    },
}


def synthetic(language, ordinal, long_only=False):
    rng = random.Random(f"{SEED}:{language}:synthetic:{ordinal}")
    target = target_chars(language, ordinal, rng, long_only)
    form = ["request", "dialogue", "quotation", "structured"][ordinal % 4]
    v, parts, seen = VOCAB[language], [], set()
    while len("\n".join(parts)) < target:
        place, obj, action, condition = [rng.choice(v[key]) for key in ("places", "objects", "actions", "conditions")]
        if language == "en":
            sentence = f"Please {action} the {obj} for the {place} {condition}."
        else:
            sentence = f"Нужно {action}: {obj}; объект: {place}; условие: {condition}."
        if sentence in seen:
            continue
        seen.add(sentence)
        if form == "dialogue":
            sentence = ("A: " if len(parts) % 2 == 0 else "B: ") + sentence
        elif form == "quotation":
            sentence = ("Note: " if language == "en" else "Заметка: ") + '"' + sentence + '"'
        elif form == "structured":
            sentence = f"[{len(parts) + 1}] location={place}; object={obj}; action={action}; condition={condition}"
        parts.append(sentence)
        if ordinal % 3 == 0 and not long_only:  # short input stays a complete sentence
            break
    text = "\n".join(parts)
    return text_record(text, language, "synthetic", {"generator": "speed_dataset.sources.synthetic",
                       "ordinal": ordinal, "seed": SEED, "license": "CC0-1.0",
                       "ground_truth": False, **({"long_only": True} if long_only else {})}, form)


def build_pool(directory, size=12000):
    from huggingface_hub import hf_hub_download
    if size not in (12000, 24000, 36000, 48000):
        raise ValueError("Pool size must be 12000, 24000, 36000 or 48000")
    directory = Path(directory)
    if (directory / "pool_manifest.json").exists():
        raise FileExistsError("Use a new directory for another pool version; old indices must remain intact")
    directory.mkdir(parents=True, exist_ok=True)
    rows, shards = [], {}
    # Prefix-stable batches: extending the pool does not rewrite existing candidates.
    wiki, synth, seen = {}, {}, set()
    for language in ("en", "ru"):
        path = hf_hub_download(WIKI_REPO, SHARDS[language], repo_type="dataset", revision=WIKI_REV)
        shards[language] = {"path": SHARDS[language], "sha256": file_hash(path)}
        wiki[language] = wiki_records(path, language, size * 3 // 8, seen)
        seen.update(r["id"] for r in wiki[language])
        synth[language], ordinal = [], 0
        while len(synth[language]) < size // 8:
            row = synthetic(language, ordinal)
            ordinal += 1
            if ordinal > 100000:
                raise ValueError("Synthetic deduplication exhausted generator budget")
            if row["id"] not in seen:
                synth[language].append(row)
                seen.add(row["id"])
    for batch in range(size // 12000):
        for language in ("en", "ru"):
            rows.extend(wiki[language][batch * 4500:(batch + 1) * 4500])
            rows.extend(synth[language][batch * 1500:(batch + 1) * 1500])
    if len({r["id"] for r in rows}) != size:
        raise ValueError("Duplicate texts detected; do not silently count them towards the pool")
    write_jsonl(directory / "pool.jsonl", rows)
    write_json(directory / "pool_manifest.json", {"version": CONTRACTS["version"], "count": size, "seed": SEED,
               "pool_sha256": file_hash(directory / "pool.jsonl"), "implementation_sha256": implementation_hash(),
               "source_shards": shards, "quality_evaluated": False, "status": "pool_only"})
    print(f"Pool ready: {size} candidates at {directory}", flush=True)


def extend_long_pool(source, directory):
    """Append a fixed 12k EN/RU long-text batch; preserve every existing byte of text."""
    from huggingface_hub import hf_hub_download
    rows, parent = validate_pool(source)
    if len(rows) + 12000 > 48000:
        raise ValueError("Extension would exceed the 48000-candidate budget")
    directory = Path(directory)
    if (directory / "pool_manifest.json").exists():
        raise FileExistsError("Use a new corpus version directory")
    directory.mkdir(parents=True, exist_ok=True)
    seen = {r["id"] for r in rows}
    shards = {}
    for language in ("en", "ru"):
        path = hf_hub_download(WIKI_REPO, SHARDS[language], repo_type="dataset", revision=WIKI_REV)
        shards[language] = {"path": SHARDS[language], "sha256": file_hash(path)}
        if shards[language] != parent["source_shards"][language]:
            raise ValueError("Parent source shard differs")
        article_ids = {r["provenance"]["article_id"] for r in rows
                       if r["source"] == "wikipedia" and r["language"] == language}
        more = wiki_records(path, language, 4500, seen, article_ids, long_only=True)
        rows.extend(more)
        seen.update(r["id"] for r in more)
        added, ordinal = 0, 0
        while added < 1500:
            row = synthetic(language, ordinal, long_only=True)
            ordinal += 1
            if ordinal > 100000:
                raise ValueError("Long synthetic deduplication exhausted")
            if row["id"] not in seen:
                rows.append(row)
                seen.add(row["id"])
                added += 1
    write_jsonl(directory / "pool.jsonl", rows)
    write_json(directory / "pool_manifest.json", {
        "version": CONTRACTS["version"], "count": len(rows), "seed": SEED,
        "pool_sha256": file_hash(directory / "pool.jsonl"), "implementation_sha256": implementation_hash(),
        "source_shards": shards, "quality_evaluated": False, "status": "pool_only",
        "parent_pool_sha256": parent["pool_sha256"], "parent_count": parent["count"],
        "extension": "12000 long EN/RU candidates, no model-specific transformation"})
    print(f"Extended pool ready: {len(rows)} candidates at {directory}", flush=True)
