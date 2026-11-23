"""scripts/eval_py150.py: the CodeXGLUE line-level record parser and the seeded subset (no data, no GPU)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "eval_py150.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("eval_py150", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# The documented record format (CodeXGLUE Code-Code/CodeCompletion-line, py150): JSON Lines,
# {"input": "<s> tok tok <EOL> tok <EOL>", "gt": "tok tok"}.
RECORD = {"input": "<s> from __future__ import absolute_import , division , print_function <EOL> "
                   "from . _ithreads import AlreadyQuit <EOL> class Quit ( object ) : <EOL>",
          "gt": "def __init__ ( self ) :"}


def test_parse_record_gives_lines_and_a_target(mod):
    ex = mod.parse_record(RECORD, 7)
    assert ex.prompt == ("from __future__ import absolute_import , division , print_function\n"
                         "from . _ithreads import AlreadyQuit\n"
                         "class Quit ( object ) :\n")
    assert ex.target == "def __init__ ( self ) :"
    assert ex.line_no == 3
    assert ex.file.endswith(":7")


def test_parse_record_cuts_gt_at_eol_and_rejects_empty(mod):
    assert mod.parse_record({"input": "<s> a = 1 <EOL>", "gt": "b = 2 <EOL> c"}, 0).target == "b = 2"
    assert mod.parse_record({"input": "<s> a = 1 <EOL>", "gt": "  "}, 0) is None


def test_load_examples_is_seeded_and_skips_blank_lines(mod, tmp_path):
    p = tmp_path / "test.json"
    rows = [{"input": f"<s> x = {i} <EOL>", "gt": f"y = {i}"} for i in range(50)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    a = mod.load_examples(p, size=10, seed=0)
    assert len(a) == 10 and a == mod.load_examples(p, size=10, seed=0)
    assert a != mod.load_examples(p, size=10, seed=1)
    assert len(mod.load_examples(p, size=999, seed=0)) == 50
