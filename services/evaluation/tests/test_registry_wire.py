"""Tests for interfaces/registry_wire.py — P5's side of the registry seam.

The first half needs only the standard library plus the dependency-free
``contracts`` package. The second half drives the *real* registry app
(``services/registry``) in-process and is skipped when it is not installed —
run it with ``pip install -e contracts -e services/registry httpx``.
"""

from __future__ import annotations

import json
import struct

import pytest

from evaluation.interfaces.contracts import AdapterKind, AdapterRef, BenchmarkName, EvalResult, InProjectMetrics
from evaluation.interfaces.registry_client import RegistryReadClient, SnapshotNotFoundError
from evaluation.interfaces.registry_wire import (
    HttpRegistryReadClient,
    eval_result_wire,
    guard_metrics,
    in_project_wire,
    promote_body,
    snapshot_from_registry,
    to_contracts_eval_result,
)
from evaluation.utils.errors import ClaspP5Error, ContractViolationError

CLUSTER = AdapterRef(name="cluster-web", version=2, kind=AdapterKind.CLUSTER, cluster_id="web")


def _bench(name: BenchmarkName, **pass_at_k: float) -> EvalResult:
    return EvalResult(
        adapter=CLUSTER,
        benchmark=name,
        pass_at_k={int(k.lstrip("k")): v for k, v in pass_at_k.items()},
        num_tasks=164,
    )


def _metrics(edit_similarity: float = 0.547, n: int = 60) -> InProjectMetrics:
    return InProjectMetrics(edit_similarity=edit_similarity, exact_match=0.2, n_examples=n)


# A document exactly as registry.storage._metadata_to_dict produces it.
REGISTRY_META = {
    "ref": {"name": "cluster-web", "version": 2, "kind": "cluster", "cluster_id": "web"},
    "hparams": {"rank": 16, "lora_alpha": 16, "dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "alpha": 1.0, "beta": 1.0},
    "privacy": {"epsilon": None, "delta": 1e-05, "noise_multiplier": None, "max_grad_norm": None},
    "aggregation": "svd_exact",
    "round": 1,
    "seed": 0,
    "sha256": "ab" * 32,
    "num_bytes": 1234,
    "source_clients": ["client-flask", "client-requests", "client-werkzeug"],
    "created_at": "2026-09-06T10:00:00+00:00",
    "contracts_version": "1.0.0",
}

#: The keys edge.promote.build_eval_result sends — the registry parses these.
EDGE_EVAL_KEYS = {"adapter", "in_project", "guard", "baseline_in_project", "baseline_noise_band", "seed"}


# ---------------------------------------------------------------------------
# Outbound
# ---------------------------------------------------------------------------
class TestGuardMetrics:
    def test_one_entry_per_benchmark_with_string_keys(self) -> None:
        guard = guard_metrics([_bench(BenchmarkName.HUMANEVAL, k1=0.31, k10=0.52), _bench(BenchmarkName.MBPP, k1=0.4)])
        assert guard == [
            {"benchmark": "HumanEval", "pass_at_k": {"1": 0.31, "10": 0.52}},
            {"benchmark": "MBPP", "pass_at_k": {"1": 0.4}},
        ]

    def test_duplicate_benchmark_refused(self) -> None:
        with pytest.raises(ContractViolationError):
            guard_metrics([_bench(BenchmarkName.HUMANEVAL, k1=0.3), _bench(BenchmarkName.HUMANEVAL, k1=0.4)])

    def test_empty_pass_at_k_refused(self) -> None:
        with pytest.raises(ContractViolationError):
            guard_metrics([EvalResult(adapter=CLUSTER, benchmark=BenchmarkName.MBPP)])


class TestInProjectWire:
    def test_perplexity_from_the_seam(self) -> None:
        assert in_project_wire(_metrics(), perplexity=2.665) == {
            "edit_similarity": 0.547, "exact_match": 0.2, "perplexity": 2.665, "n_examples": 60,
        }

    def test_missing_perplexity_is_refused_not_invented(self) -> None:
        with pytest.raises(ContractViolationError, match="perplexity"):
            in_project_wire(_metrics())

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_bad_perplexity_refused(self, bad: float) -> None:
        with pytest.raises(ContractViolationError):
            in_project_wire(_metrics(), perplexity=bad)

    def test_zero_examples_refused(self) -> None:
        with pytest.raises(ContractViolationError):
            in_project_wire(_metrics(n=0), perplexity=2.0)


class TestEvalResultWire:
    def _wire(self, **kw):
        return eval_result_wire(
            CLUSTER,
            in_project_wire(_metrics(), perplexity=2.665),
            guard=guard_metrics([_bench(BenchmarkName.HUMANEVAL, k1=0.31)]),
            baseline_in_project=in_project_wire(_metrics(0.5337), perplexity=2.7),
            **kw,
        )

    def test_same_keys_as_the_edge_lane_sends(self) -> None:
        assert set(self._wire()) == EDGE_EVAL_KEYS

    def test_is_json_serialisable(self) -> None:
        assert json.loads(json.dumps(self._wire(baseline_noise_band=0.004)))["baseline_noise_band"] == 0.004

    def test_parses_into_the_frozen_contract(self) -> None:
        import contracts

        parsed = to_contracts_eval_result(self._wire(seed=3))
        assert isinstance(parsed, contracts.EvalResult)
        assert parsed.adapter.kind is contracts.AdapterKind.CLUSTER
        assert parsed.guard[0].pass_at_k == {1: 0.31}
        assert parsed.baseline_in_project.edit_similarity == 0.5337
        assert parsed.contracts_version == "1.0.0"

    def test_negative_band_refused(self) -> None:
        with pytest.raises(ContractViolationError):
            self._wire(baseline_noise_band=-0.1)

    def test_promote_body_shape(self) -> None:
        body = promote_body(self._wire(), guard_metrics([_bench(BenchmarkName.HUMANEVAL, k1=0.32)]))
        assert set(body) == {"eval", "baseline_guard"}
        assert body["baseline_guard"][0]["benchmark"] == "HumanEval"


# ---------------------------------------------------------------------------
# Inbound
# ---------------------------------------------------------------------------
class TestSnapshotFromRegistry:
    def test_maps_registry_metadata(self) -> None:
        snap = snapshot_from_registry(REGISTRY_META)
        assert snap.adapter == CLUSTER
        assert snap.round_number == 1
        assert snap.artifact_sha256 == "ab" * 32
        assert snap.artifact_path == "/adapters/cluster-web/versions/2/file"

    def test_client_adapter_without_round(self) -> None:
        meta = dict(REGISTRY_META, ref={"name": "client-flask", "version": 1, "kind": "client", "cluster_id": None}, round=None)
        snap = snapshot_from_registry(meta)
        assert snap.adapter.kind is AdapterKind.CLIENT and snap.round_number == 0

    def test_rejects_non_metadata(self) -> None:
        with pytest.raises(ContractViolationError):
            snapshot_from_registry({"name": "x"})


class _FakeResponse:
    def __init__(self, status_code: int, payload) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, routes: dict[str, tuple[int, object]]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url: str) -> _FakeResponse:
        self.calls.append(url)
        status, payload = self.routes.get(url, (404, {"detail": "not found"}))
        return _FakeResponse(status, payload)


class TestHttpRegistryReadClientOffline:
    def _client(self, routes):
        return HttpRegistryReadClient("http://registry:8004/", session=_FakeSession(routes))

    def test_satisfies_the_read_protocol(self) -> None:
        assert isinstance(self._client({}), RegistryReadClient)

    def test_list_filters_kind_and_sorts(self) -> None:
        v1 = dict(REGISTRY_META, ref=dict(REGISTRY_META["ref"], version=1))
        client = self._client({"http://registry:8004/adapters/cluster-web/versions": (200, {"versions": [REGISTRY_META, v1]})})
        assert [s.adapter.version for s in client.list_versions("cluster-web", AdapterKind.CLUSTER)] == [1, 2]
        assert client.list_versions("cluster-web", AdapterKind.CLIENT) == []

    def test_unknown_adapter(self) -> None:
        client = self._client({})
        assert client.list_versions("nope", AdapterKind.CLUSTER) == []
        with pytest.raises(SnapshotNotFoundError):
            client.latest("nope", AdapterKind.CLUSTER)
        with pytest.raises(SnapshotNotFoundError):
            client.active("nope")

    def test_server_error_is_not_swallowed(self) -> None:
        client = self._client({"http://registry:8004/adapters/x/active": (500, {})})
        with pytest.raises(ClaspP5Error, match="500"):
            client.active("x")

    def test_unreachable_registry_is_a_clean_error(self) -> None:
        with pytest.raises(ClaspP5Error, match="unreachable"):
            HttpRegistryReadClient("http://127.0.0.1:9", timeout=0.5).active("cluster-web")


# ---------------------------------------------------------------------------
# Against the real State Registry app (services/registry), in-process
# ---------------------------------------------------------------------------
def _safetensors_blob(values=(0.1, 0.2, 0.3, 0.4)) -> bytes:
    data = struct.pack(f"<{len(values)}f", *values)
    header = json.dumps({"lora_A": {"dtype": "F32", "shape": [len(values)], "data_offsets": [0, len(data)]}}).encode()
    return struct.pack("<Q", len(header)) + header + data


@pytest.fixture
def registry(tmp_path, monkeypatch):
    pytest.importorskip("fastapi.testclient")
    appmod = pytest.importorskip("registry.app")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("CLASP_REGISTRY_DATA", str(tmp_path / "registry"))
    appmod._store = None
    with TestClient(appmod.app) as client:
        for version, value in ((1, 0.1), (2, 0.2)):
            meta = {"kind": "cluster", "cluster_id": "web", "round": version, "aggregation": "svd_exact",
                    "source_clients": ["client-flask"]}
            resp = client.post(
                "/adapters/cluster-web/versions",
                files={"file": ("a.safetensors", _safetensors_blob((value,) * 4), "application/octet-stream")},
                data={"meta": json.dumps(meta)},
            )
            assert resp.status_code == 201, resp.text
        yield client
    appmod._store = None


class TestAgainstRealRegistry:
    def test_reads_versions_active_and_metadata(self, registry) -> None:
        reader = HttpRegistryReadClient("", session=registry)
        versions = reader.list_versions("cluster-web", AdapterKind.CLUSTER)
        assert [s.adapter.version for s in versions] == [1, 2]
        assert reader.active("cluster-web").adapter.version == 2  # saves auto-activate
        assert reader.latest("cluster-web", AdapterKind.CLUSTER) == versions[-1]
        one = reader.load_metadata(AdapterRef("cluster-web", 1, AdapterKind.CLUSTER, "web"))
        assert len(one.artifact_sha256) == 64
        assert registry.get(one.artifact_path).status_code == 200

    def _promote(self, registry, *, candidate_pass1: float, candidate_edit_sim: float):
        reader = HttpRegistryReadClient("", session=registry)
        active = reader.active("cluster-web").adapter
        wire = eval_result_wire(
            active,
            in_project_wire(_metrics(candidate_edit_sim), perplexity=2.665),
            guard=guard_metrics([_bench(BenchmarkName.HUMANEVAL, k1=candidate_pass1)]),
            baseline_in_project=in_project_wire(_metrics(0.5337), perplexity=2.7),
            baseline_noise_band=0.0,
        )
        baseline_guard = guard_metrics([_bench(BenchmarkName.HUMANEVAL, k1=0.30)])
        resp = registry.post("/adapters/cluster-web/promote", json=promote_body(wire, baseline_guard))
        assert resp.status_code == 200, resp.text
        return resp.json(), reader

    def test_d5_promotes_on_p5_metrics(self, registry) -> None:
        decision, reader = self._promote(registry, candidate_pass1=0.30, candidate_edit_sim=0.547)
        assert decision["action"] == "promote", decision["reason"]
        assert reader.active("cluster-web").adapter.version == 2

    def test_d5_rolls_back_when_the_p5_guard_drops(self, registry) -> None:
        decision, reader = self._promote(registry, candidate_pass1=0.25, candidate_edit_sim=0.547)
        assert decision["action"] == "rollback"
        assert "HumanEval pass@1 dropped" in decision["reason"]
        assert reader.active("cluster-web").adapter.version == 1
