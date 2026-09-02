"""Packing + budget-planner tests (P1 Edge, W3 · E3.1/E3.2).

No GPU and no model download: a stub tokenizer stands in for the real one, so
these run in CI on the 6-package matrix (D9).

The determinism tests are load-bearing. E3.2's deliverable is a *repeatable*
config, and that claim is only meaningful if the same partition packs into the
same blocks every time — an unsorted directory walk would silently reshuffle
every block boundary between Windows and the Linux GPU box.
"""
import json

import pytest

from edge.chunking import (
    chunk_tokens,
    discover_clients,
    load_files,
    order_hash,
    pack_client,
    plan_budgets,
    tokenize_repo,
)

EOS = 32014


class StubTokenizer:
    """One token per character, so token counts are predictable in tests."""

    eos_token_id = EOS

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 1000 for c in text]}


@pytest.fixture
def client_dir(tmp_path):
    """A minimal client partition: repo/ + held_out/, mirroring P5's layout."""
    repo = tmp_path / "client-x" / "repo"
    held = tmp_path / "client-x" / "held_out"
    (repo / "pkg").mkdir(parents=True)
    held.mkdir(parents=True)
    (repo / "__init__.py").write_text("a" * 500, encoding="utf-8")
    (repo / "zebra.py").write_text("b" * 500, encoding="utf-8")
    (repo / "pkg" / "mod.py").write_text("c" * 500, encoding="utf-8")
    (held / "test_held.py").write_text("d" * 200, encoding="utf-8")
    return tmp_path / "client-x"


# --- packing ---------------------------------------------------------------

def test_load_files_is_sorted_not_filesystem_order(client_dir):
    """The whole determinism story rests on this. __init__.py must come first
    even though a raw rglob does not put it there."""
    names = [rel for rel, _ in load_files(client_dir / "repo")]
    assert names == sorted(names)
    assert names[0] == "__init__.py"


def test_load_files_uses_posix_separators(client_dir):
    """Manifest hashes must match across OSes, so no backslashes on Windows."""
    names = [rel for rel, _ in load_files(client_dir / "repo")]
    assert "pkg/mod.py" in names
    assert not any("\\" in n for n in names)


def test_tokenize_repo_separates_files_with_eos(client_dir):
    out = tokenize_repo(client_dir / "repo", StubTokenizer())
    # 3 files x 500 chars + one EOS each
    assert len(out["tokens"]) == 3 * 501
    assert out["tokens"].count(EOS) == 3
    assert out["tokens"][-1] == EOS
    assert len(out["file_order"]) == 3


def test_order_hash_detects_reordering():
    a = order_hash(["a.py", "b.py"])
    assert a != order_hash(["b.py", "a.py"])   # order, not just membership
    assert a == order_hash(["a.py", "b.py"])   # and it is stable


def test_order_hash_is_not_length_only():
    """A rename that preserves file COUNT must still move the hash."""
    assert order_hash(["a.py", "b.py"]) != order_hash(["a.py", "c.py"])


def test_chunk_tokens_drop_last_yields_only_full_blocks():
    chunks = chunk_tokens(list(range(2500)), seq_len=1024, drop_last=True)
    assert len(chunks) == 2
    assert all(len(c) == 1024 for c in chunks)


def test_chunk_tokens_keep_last_preserves_every_token():
    tokens = list(range(2500))
    chunks = chunk_tokens(tokens, seq_len=1024, drop_last=False)
    assert sum(len(c) for c in chunks) == len(tokens)
    assert [t for c in chunks for t in c] == tokens


def test_chunk_tokens_shorter_than_one_block():
    """flask's held-out split is 514 tokens. drop_last would erase it, which is
    why eval packs with drop_last=False."""
    assert chunk_tokens(list(range(514)), seq_len=1024, drop_last=True) == []
    assert len(chunk_tokens(list(range(514)), seq_len=1024, drop_last=False)) == 1


def test_chunk_tokens_exact_multiple_has_no_phantom_tail():
    tokens = list(range(2048))
    assert len(chunk_tokens(tokens, seq_len=1024, drop_last=False)) == 2


def test_pack_client_is_deterministic(client_dir):
    a, _ = pack_client(client_dir, StubTokenizer(), seq_len=128)
    b, _ = pack_client(client_dir, StubTokenizer(), seq_len=128)
    assert a["order_sha256"] == b["order_sha256"]
    assert a["chunks"] == b["chunks"]


def test_pack_client_train_and_held_out_stay_separate(client_dir):
    """D5's in-project eval set must never appear in training data."""
    train, held = pack_client(client_dir, StubTokenizer(), seq_len=128)
    assert "test_held.py" not in train["file_order"]
    assert held["file_order"] == ["test_held.py"]
    assert all(len(c) == 128 for c in train["chunks"])


def test_pack_client_rejects_a_client_too_small_to_train(client_dir):
    with pytest.raises(ValueError, match="less than one"):
        pack_client(client_dir, StubTokenizer(), seq_len=100_000)


def test_discover_clients_skips_underscore_dirs(tmp_path):
    for cluster, client in [("web", "client-a"), ("web", "_extra"), ("_scratch", "client-b")]:
        (tmp_path / cluster / client / "repo").mkdir(parents=True)
    found = discover_clients(tmp_path)
    assert set(found) == {"web/client-a"}


# --- budgeting -------------------------------------------------------------

REAL_COUNTS = {
    "scientific/client-numpy": 1286,
    "scientific/client-pandas": 1687,
    "scientific/client-scikit-learn": 1696,
    "web/client-flask": 80,
    "web/client-requests": 52,
    "web/client-werkzeug": 170,
}


def test_uniform_policy_reproduces_the_imbalance():
    """The baseline this whole exercise exists to replace: ~33x epoch spread."""
    plan = plan_budgets(REAL_COUNTS, total_chunks=1200, policy="uniform", max_epochs=99)
    eps = [r["epochs"] for r in plan.values()]
    assert max(eps) / min(eps) > 25


def test_sqrt_policy_compresses_the_spread():
    plan = plan_budgets(REAL_COUNTS, total_chunks=1200, policy="sqrt")
    eps = [r["epochs"] for r in plan.values()]
    assert max(eps) / min(eps) < 8
    # and the small clients are the ones that gained
    assert plan["web/client-requests"]["epochs"] > plan["scientific/client-numpy"]["epochs"]


def test_linear_policy_equalizes_epochs():
    plan = plan_budgets(REAL_COUNTS, total_chunks=100_000, policy="linear", max_epochs=99)
    eps = [r["epochs"] for r in plan.values()]
    assert max(eps) - min(eps) < 0.05


def test_budget_respects_the_total_spend():
    """The pitch to the team is 'same GPU-hours, redistributed'. Hold us to it."""
    plan = plan_budgets(REAL_COUNTS, total_chunks=1200, policy="sqrt", max_epochs=99)
    assert sum(r["chunk_budget"] for r in plan.values()) == pytest.approx(1200, abs=6)


def test_max_epochs_caps_tiny_clients():
    plan = plan_budgets({"tiny": 10, "huge": 10_000}, total_chunks=1200,
                        policy="sqrt", max_epochs=2.0)
    assert plan["tiny"]["epochs"] <= 2.0


def test_min_chunks_prevents_a_starved_client():
    """A client with almost no budget still carries aggregation weight in D2,
    so it must not be reduced to a meaningless update."""
    plan = plan_budgets({"tiny": 30, "huge": 100_000}, total_chunks=200,
                        policy="linear", min_chunks=20)
    assert plan["tiny"]["chunk_budget"] >= 20


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="policy must be one of"):
        plan_budgets(REAL_COUNTS, policy="magic")


def test_plan_is_json_serializable():
    """It goes straight into the run manifest (D9)."""
    json.dumps(plan_budgets(REAL_COUNTS))


def test_plan_is_deterministic():
    assert plan_budgets(REAL_COUNTS) == plan_budgets(REAL_COUNTS)
