"""D7 in the registry (U3): what each client's run has spent, recorded with the version.

The cluster publishes ``client_privacy`` beside a cluster version: one row per
aggregated client with the run's cumulative epsilon the client's own ledger
reported. The registry writes those rows in the transaction that registers the
version and serves them from ``GET /privacy/ledger``.
"""
from __future__ import annotations

import json

import pytest
from sqlalchemy import inspect

from registry import db

from .conftest import forget_database  # noqa: F401 - documents the volume helper is unused here
from .test_composite import make_adapter
from .test_composite_api import HP, _save


def _row(client_id, epsilon_round, cumulative, rounds):
    return {"client_id": client_id, "epsilon_round": epsilon_round,
            "epsilon_cumulative": cumulative, "rounds_accounted": rounds}


def _publish(client, name="cluster-web", cluster_id="web", seed=1, round=1, rows=None, **meta):
    extra = {"client_privacy": rows} if rows is not None else {}
    return _save(client, name, make_adapter(seed), kind="cluster", aggregation="svd_exact",
                 cluster_id=cluster_id, round=round, set_active=False, **extra, **meta)


def _ledger(client, **params):
    r = client.get("/privacy/ledger", params=params)
    assert r.status_code == 200, r.text
    return r.json()


# ---- schema ----------------------------------------------------------------------
def test_the_table_and_its_index_exist(tmp_path):
    engine = db.make_engine(f"sqlite:///{(tmp_path / 'r.db').as_posix()}")
    db.create_all(engine)
    insp = inspect(engine)
    assert "privacy_ledger" in insp.get_table_names()
    cols = {c["name"]: c for c in insp.get_columns("privacy_ledger")}
    assert set(cols) == {"id", "client_id", "cluster_id", "adapter_version_id", "round",
                         "epsilon_round", "epsilon_cumulative", "rounds_accounted", "delta",
                         "recorded_at"}
    for required in ("client_id", "cluster_id", "epsilon_cumulative"):
        assert cols[required]["nullable"] is False
    assert any(set(u["column_names"]) == {"client_id", "adapter_version_id"}
               for u in insp.get_unique_constraints("privacy_ledger"))
    assert any(i["column_names"] == ["client_id", "recorded_at"]
               for i in insp.get_indexes("privacy_ledger"))


# ---- recording ---------------------------------------------------------------------
def test_an_upload_with_client_privacy_creates_one_row_per_client(client):
    written = _publish(client, privacy={"epsilon": 3.9, "epsilon_cumulative": 3.9}, rows=[
        _row("client-flask", 3.9, 3.9, 1), _row("client-requests", 3.2, 3.2, 1)])
    body = _ledger(client)
    assert body["budget"] == 8.0 and body["delta"] == 1e-5
    rows = body["rows"]
    assert [r["client_id"] for r in rows] == ["client-flask", "client-requests"]
    flask = rows[0]
    assert flask["cluster_id"] == "web" and flask["adapter"] == "cluster-web"
    assert flask["version"] == written["ref"]["version"] and flask["round"] == 1
    assert flask["epsilon_round"] == 3.9 and flask["epsilon_cumulative"] == 3.9
    assert flask["rounds_accounted"] == 1 and flask["delta"] == 1e-5 and flask["recorded_at"]


def test_a_meta_without_client_privacy_creates_no_rows_and_still_succeeds(client):
    _publish(client)
    assert _ledger(client)["rows"] == []


def test_a_client_trained_without_dp_gets_no_row(client):
    _publish(client, rows=[_row("client-flask", 3.9, 3.9, 1),
                           _row("client-plain", None, None, None)])
    assert [r["client_id"] for r in _ledger(client)["rows"]] == ["client-flask"]


def test_a_second_round_adds_rows_and_the_first_stay(client):
    _publish(client, seed=1, round=1, rows=[_row("client-flask", 3.9, 3.9, 1)])
    _publish(client, seed=2, round=2, rows=[_row("client-flask", 3.9, 7.8, 2)])
    rows = _ledger(client)["rows"]
    assert [(r["version"], r["epsilon_cumulative"], r["rounds_accounted"]) for r in rows] == [
        (1, 3.9, 1), (2, 7.8, 2)]


def test_the_route_filters_by_cluster(client):
    _publish(client, "cluster-web", "web", seed=1, rows=[_row("client-flask", 3.9, 3.9, 1)])
    _publish(client, "cluster-scientific", "scientific", seed=2,
             rows=[_row("client-numpy", 2.0, 2.0, 1)])
    assert [r["client_id"] for r in _ledger(client, cluster_id="web")["rows"]] == ["client-flask"]
    assert [r["client_id"] for r in _ledger(client, cluster_id="scientific")["rows"]] == [
        "client-numpy"]
    assert _ledger(client, cluster_id="nope")["rows"] == []
    assert len(_ledger(client)["rows"]) == 2


@pytest.mark.parametrize("bad", [
    "not a list",
    [{"epsilon_cumulative": 3.9}],                                   # no client
    [{"client_id": "client-flask", "epsilon_cumulative": "lots"}],
    [{"client_id": "client-flask", "epsilon_cumulative": -1.0}],
    [{"client_id": "client-flask", "epsilon_cumulative": 3.9, "rounds_accounted": 0}],
    [{"client_id": "client-flask", "epsilon_round": 4.0, "epsilon_cumulative": 3.9}],
    [_row("client-flask", 3.9, 3.9, 1), _row("client-flask", 3.9, 3.9, 1)],   # named twice
])
def test_malformed_client_privacy_is_422_and_nothing_is_saved(client, bad):
    r = client.post(
        "/adapters/cluster-web/versions",
        files={"file": ("a.safetensors", make_adapter(1), "application/octet-stream")},
        data={"meta": json.dumps({"hparams": HP, "kind": "cluster", "aggregation": "svd_exact",
                                  "cluster_id": "web", "client_privacy": bad})},
    )
    assert r.status_code == 422, r.text
    assert "client_privacy" in r.text
    assert client.get("/adapters").json()["adapters"] == []
    assert _ledger(client)["rows"] == []


def test_rows_are_written_with_the_version_or_not_at_all(client, monkeypatch):
    import registry.app as appmod

    repo = appmod.get_store().repo

    def boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(repo, "add_privacy_rows", boom)
    with pytest.raises(RuntimeError):
        _publish(client, rows=[_row("client-flask", 3.9, 3.9, 1)])
    monkeypatch.undo()
    assert client.get("/adapters").json()["adapters"] == []
    assert _ledger(client)["rows"] == []


def test_a_garbage_collected_version_keeps_its_spend(client):
    """Deleting old weights does not give the budget back."""
    import registry.app as appmod

    _publish(client, seed=1, round=1, rows=[_row("client-flask", 3.9, 3.9, 1)])
    _publish(client, seed=2, round=2, rows=[_row("client-flask", 3.9, 7.8, 2)])
    appmod.get_store().repo.delete_version("cluster-web", 1)
    rows = _ledger(client)["rows"]
    assert [r["epsilon_cumulative"] for r in rows] == [3.9, 7.8]
    assert rows[0]["version"] is None and rows[1]["version"] == 2
