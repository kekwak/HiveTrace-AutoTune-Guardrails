"""Локальный прогон тестов: `u1` — только тесты У1, `u2` — только наши, `all` — все (как unittest discover)."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def flatten(suite):
    for item in suite:
        yield from flatten(item) if isinstance(item, unittest.TestSuite) else [item]


which = sys.argv[1] if len(sys.argv) > 1 else "all"
tests = flatten(unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT / "tests")))
keep = {"u1": lambda t: "test_u2_" not in t.id(), "u2": lambda t: "test_u2_" in t.id(),
        "all": lambda t: True}[which]
result = unittest.TextTestRunner(verbosity=1).run(unittest.TestSuite(t for t in tests if keep(t)))
sys.exit(not result.wasSuccessful())
