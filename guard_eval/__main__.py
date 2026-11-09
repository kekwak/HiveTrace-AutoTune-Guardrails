"""guard_eval: качество и скорость guardrail-моделей, включая динамически квантованные версии.

python -m guard_eval check                                   # что поддерживает текущая GPU
python -m guard_eval run     --model koala --quant torchao_int8_dyn
python -m guard_eval speed   --model koala --quant torchao_int8_dyn
python -m guard_eval compare --model koala --a none --b torchao_int8_dyn [--speed]
python -m guard_eval compare --spec-a specs/x.yaml --spec-b specs/y.yaml
python -m guard_eval summary                                 # сводка всех сравнений
"""

import argparse
import json
import sys

from . import quant
from .data import DEFAULT_DATASET, load_dataset
from .spec import MODELS_DIR, load_spec, resolve_dtype, run_label, with_quant


def _device(args):
    import torch
    if args.device:
        return args.device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _dataset(args):
    df, origin = load_dataset(args.data, revision=args.data_revision, benchmarks=args.benchmarks,
                              limit_per_split=args.limit_per_split)
    print(f"датасет: {origin}")
    return df, origin


def _ensure_quality(spec, args, df):
    from .runner import run_quality
    return run_quality(spec, df, _device(args), save_every=args.save_every)


def _ensure_speed(spec, args, run):
    from .speed import measure
    existing = run.speed()
    if existing and existing.get("status") == "complete" and not args.force_speed:
        print(f"[{run_label(spec)}] скорость уже измерена: {run.speed_path}")
        return existing
    print(f"[{run_label(spec)}] замер скорости")
    result = measure(spec, _device(args), warmup=args.speed_warmup, blocks=args.speed_blocks, per_block=args.speed_per_block)
    run.save_speed(result)
    return result


def _frame(run, df):
    preds = run.predictions()[["text_hash", "p_unsafe", "error"]]
    return df.merge(preds, on="text_hash", how="left")


def cmd_check(args):
    from . import environment
    device = _device(args)
    env = environment.describe(device)
    print(json.dumps({k: env[k] for k in env if k != "packages"}, ensure_ascii=False, indent=2))
    print("пакеты:", json.dumps(env["packages"], ensure_ascii=False))
    for method in quant.METHODS[1:]:
        reason = quant.check_support(method, device)
        print(f"{method:18} {'поддерживается' if not reason else 'нет: ' + reason}")
    if device.startswith("cuda"):
        import torch
        # та же точность, что выберет dtype: auto — на T4 это fp16, а не bf16
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(torch.device(device)) >= (8, 0) else torch.float16
        print(f"smoke на маленькой сети ({dtype}):", json.dumps(quant.smoke(device, dtype), ensure_ascii=False, indent=2))


def cmd_run(args):
    from .metrics import evaluate
    from .report import write_single
    spec = resolve_dtype(with_quant(load_spec(args.model, args.set), args.quant), _device(args))
    df, origin = _dataset(args)
    run = _ensure_quality(spec, args, df)
    if run.manifest().get("status") in ("unsupported", "failed"):
        print(f"{run_label(spec)}: прогон не выполнен — {run.manifest().get('error')}")
        return 2
    if args.speed:
        _ensure_speed(spec, args, run)
    tables = evaluate(_frame(run, df), spec["threshold"])
    out = write_single(run, tables, origin, spec)
    print(tables["overall"][["aggregate", "fnr", "fpr", "macro_f1"]].round(4).to_string(index=False))
    print(f"отчёт: {out / 'summary.md'}")
    return 0


def cmd_speed(args):
    from .runner import prepare_run
    spec = resolve_dtype(with_quant(load_spec(args.model, args.set), args.quant), _device(args))
    run, _ = prepare_run(spec, _device(args))
    result = _ensure_speed(spec, args, run)
    return 0 if result.get("status") == "complete" else 2


def cmd_compare(args):
    from .metrics import compare
    from .report import write_compare
    if args.spec_a or args.spec_b:
        spec_a = load_spec(args.spec_a, args.set + args.set_a)
        spec_b = load_spec(args.spec_b, args.set + args.set_b)
    else:
        spec_a = with_quant(load_spec(args.model, args.set + args.set_a), args.a)
        spec_b = with_quant(load_spec(args.model, args.set + args.set_b), args.b)
    spec_a, spec_b = resolve_dtype(spec_a, _device(args)), resolve_dtype(spec_b, _device(args))
    if spec_a == spec_b:
        raise SystemExit("A и B совпадают")
    df, origin = _dataset(args)
    run_a = _ensure_quality(spec_a, args, df)
    run_b = _ensure_quality(spec_b, args, df)
    for spec, run in ((spec_a, run_a), (spec_b, run_b)):
        if run.manifest().get("status") in ("unsupported", "failed"):
            print(f"{run_label(spec)}: прогон не выполнен ({run.manifest().get('error')}) — сравнение невозможно")
            return 2
    if args.speed:
        _ensure_speed(spec_a, args, run_a)
        _ensure_speed(spec_b, args, run_b)
    frame = _frame(run_a, df).rename(columns={"p_unsafe": "p_a", "error": "error_a"}).merge(
        run_b.predictions()[["text_hash", "p_unsafe"]].rename(columns={"p_unsafe": "p_b"}), on="text_hash", how="left")
    tables, rows = compare(frame, (spec_a["threshold"], spec_b["threshold"]))
    out = write_compare(run_a, run_b, tables, rows, origin, spec_a, spec_b)
    o = tables["overall"]
    print(o[["aggregate", "flip_rate", "flip_rate_lo", "flip_rate_hi", "delta_fnr", "delta_fpr", "delta_macro_f1"]]
          .round(4).to_string(index=False))
    print(f"отчёт: {out / 'summary.md'}")
    return 0


def cmd_summary(args):
    from .report import write_overview
    out, n = write_overview()
    print(f"сравнений: {n}; сводка: {out}")
    return 0


def main(argv=None):
    models = ", ".join(sorted(p.stem for p in MODELS_DIR.glob("*.yaml")))
    p = argparse.ArgumentParser(prog="guard_eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, model=True):
        if model:
            sp.add_argument("--model", help=f"псевдоним ({models}) или путь к YAML")
            sp.add_argument("--quant", choices=quant.METHODS, help="метод квантования (по умолчанию из spec)")
        sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="переопределить поле spec, напр. dtype=float32 или batch_size=1")
        sp.add_argument("--device", help="cuda:0 / cpu (по умолчанию cuda, если доступна)")
        sp.add_argument("--data", default=DEFAULT_DATASET, help="HF-датасет или локальный путь к tails_v1")
        sp.add_argument("--data-revision")
        sp.add_argument("--benchmarks", nargs="+")
        sp.add_argument("--limit-per-split", type=int, help="подвыборка для быстрой проверки")
        sp.add_argument("--save-every", type=int, default=512)
        sp.add_argument("--force-speed", action="store_true", help="перемерить скорость, даже если есть")
        sp.add_argument("--speed-warmup", type=int, default=20)
        sp.add_argument("--speed-blocks", type=int, default=3)
        sp.add_argument("--speed-per-block", type=int, default=100)

    sp = sub.add_parser("check", help="поддержка методов квантования на текущем устройстве")
    sp.add_argument("--device")
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("run", help="прогон качества одной конфигурации (+ отчёт)")
    common(sp)
    sp.add_argument("--speed", action="store_true", help="также замерить скорость")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("speed", help="замер скорости по бакетам")
    common(sp)
    sp.set_defaults(func=cmd_speed)

    sp = sub.add_parser("compare", help="сравнение двух конфигураций; недостающие прогоны досчитываются")
    common(sp)
    sp.add_argument("--a", default="none", choices=quant.METHODS, help="quant для A (эталон)")
    sp.add_argument("--b", choices=quant.METHODS, help="quant для B (по умолчанию как в spec)")
    sp.add_argument("--set-a", action="append", default=[], metavar="KEY=VALUE", help="переопределение только для A")
    sp.add_argument("--set-b", action="append", default=[], metavar="KEY=VALUE", help="переопределение только для B")
    sp.add_argument("--spec-a"), sp.add_argument("--spec-b")
    sp.add_argument("--speed", action="store_true", help="также замерить и сравнить скорость")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("summary", help="сводка всех сравнений в reports/SUMMARY.md")
    sp.set_defaults(func=cmd_summary)

    args = p.parse_args(argv)
    if args.cmd in ("run", "speed") and not args.model:
        p.error("нужен --model")
    if args.cmd == "compare" and not ((args.spec_a and args.spec_b) or (args.model and (args.b or args.set_b or args.set_a))):
        p.error("нужно --model и --b (или --set-a/--set-b), либо --spec-a и --spec-b")
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
