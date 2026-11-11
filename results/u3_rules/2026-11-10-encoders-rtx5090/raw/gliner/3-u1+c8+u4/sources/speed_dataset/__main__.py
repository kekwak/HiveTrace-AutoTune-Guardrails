"""python -m speed_dataset {pool,index,finalize,package}"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Общий CPU-корпус скорости, без инференса")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pool")
    p.add_argument("directory")
    p.add_argument("--size", type=int, default=12000, choices=[12000, 24000, 36000, 48000])
    p = sub.add_parser("index")
    p.add_argument("directory")
    p.add_argument("--model", required=True)
    p.add_argument("--reuse", help="Previous pool directory; reuse only verified identical text/contract/assets")
    p = sub.add_parser("finalize")
    p.add_argument("directory")
    p = sub.add_parser("package")
    p.add_argument("directory")
    p = sub.add_parser("import-index")
    p.add_argument("directory")
    p.add_argument("archive")
    p = sub.add_parser("extend-long")
    p.add_argument("source")
    p.add_argument("directory")
    args = parser.parse_args()
    if args.command == "pool":
        from .sources import build_pool
        build_pool(args.directory, args.size)
    elif args.command == "index":
        from .pipeline import index_pool
        index_pool(args.directory, args.model, args.reuse)
    elif args.command == "finalize":
        from .pipeline import finalize
        report = finalize(args.directory)
        return 0 if report["status"] == "ready" else 2
    elif args.command == "package":
        from .package import package
        package(args.directory)
    elif args.command == "import-index":
        from .pipeline import import_index
        import_index(args.directory, args.archive)
    else:
        from .sources import extend_long_pool
        extend_long_pool(args.source, args.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
