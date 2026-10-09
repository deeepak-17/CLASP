"""Tests for eval_harness/lineage.py — adapter lineage across edge rounds."""

from __future__ import annotations

from evaluation.eval_harness.lineage import build_lineage
from evaluation.utils.io_utils import read_json
from evaluation.utils.paths import project_paths


def _rounds() -> list[dict]:
    root = project_paths().eval_results / "edge_rounds"
    return [read_json(root / "round1_manifest.json"), read_json(root / "round2_d3_manifest.json")]


def test_real_rounds_graph_shape() -> None:
    lineage = build_lineage(_rounds())
    kinds = [n["kind"] for n in lineage["nodes"]]
    assert kinds.count("client") == 12 and kinds.count("cluster") == 4 and kinds.count("composite") == 12
    assert lineage["rounds"] == [1, 2]


def test_round_two_clients_trained_on_the_registry_version() -> None:
    lineage = build_lineage(_rounds())
    nodes = {n["id"]: n for n in lineage["nodes"]}
    assert nodes["r2:web/client-flask"]["trained_on"] == "base + 0.5·cluster-web v2"
    assert {"source": "registry:cluster-web@v2", "target": "r2:web/client-flask", "kind": "trained_on"} in lineage[
        "edges"
    ]
    assert nodes["registry:cluster-web@v2"]["sha256"].startswith("9758be92")


def test_registry_cluster_aggregates_round_one_uploads() -> None:
    lineage = build_lineage(_rounds())
    sources = {e["source"] for e in lineage["edges"] if e["target"] == "registry:cluster-scientific@v2"}
    assert sources == {
        "r1:scientific/client-numpy",
        "r1:scientific/client-pandas",
        "r1:scientific/client-scikit-learn",
    }
