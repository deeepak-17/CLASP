"""Tests for the P1–P4 integration seam: contracts, mocks and validation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from interfaces.cluster_client import (
    ClusterPlanConsumer,
    MockClusterClient,
    PlanAcceptance,
)
from interfaces.contracts import (
    CONTRACT_VERSION,
    AdapterKind,
    AdapterRef,
    BenchmarkName,
    EvalResult,
    PartitionManifest,
    PartitionShard,
    PartitionStrategyName,
    SnapshotMetadata,
)
from interfaces.edge_client import (
    EdgeInferenceClient,
    GenerationRequest,
    MockEdgeInferenceClient,
)
from interfaces.edge_transformers_adapter import (
    LoraComposition,
    TransformersEdgeInferenceClient,
    register_transformers_edge_client,
)
from interfaces.registry_client import (
    MockRegistryClient,
    RegistryReadClient,
    SnapshotNotFoundError,
)
from interfaces.security_client import (
    NullPrivacyAccountant,
    PrivacyAccountant,
    PrivacyBudget,
)
from interfaces.validation import (
    SCHEMA_REGISTRY,
    jsonschema_available,
    load_schema,
    validate_document,
    validate_file,
)
from utils.errors import ClaspP5Error, ContractViolationError, DependencyError
from utils.timing import utc_timestamp


def make_shard(client_id: str = "client-alpha", num_files: int = 4) -> PartitionShard:
    return PartitionShard(
        client_id=client_id,
        cluster_id=client_id.removeprefix("client-"),
        project_label=client_id.removeprefix("client-").title(),
        num_files=num_files,
        num_lines=num_files * 40,
        num_code_lines=num_files * 30,
        total_bytes=num_files * 900,
        files_path=f"datasets/partitions/{client_id}.jsonl",
        content_sha256="a" * 64,
    )


def make_manifest(shards: list[PartitionShard] | None = None) -> PartitionManifest:
    shards = shards or [make_shard("client-alpha", 4), make_shard("client-beta", 6)]
    return PartitionManifest(
        manifest_version="1.0.0",
        strategy=PartitionStrategyName.PROJECT_LEVEL,
        seed=7,
        created_at=utc_timestamp(),
        corpus_path="datasets/processed/corpus.jsonl",
        corpus_sha256="b" * 64,
        num_clients=len(shards),
        num_clusters=len({s.cluster_id for s in shards}),
        total_files=sum(s.num_files for s in shards),
        shards=shards,
        cluster_ids=sorted({s.cluster_id for s in shards}),
    )


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------
class TestAdapterRef:
    def test_uri_format(self) -> None:
        ref = AdapterRef(name="flask", version=3, kind=AdapterKind.CLUSTER, cluster_id="flask")
        assert ref.uri == "cluster/flask@3"

    def test_round_trips(self) -> None:
        ref = AdapterRef(name="dev-1", version=0, kind=AdapterKind.CLIENT)
        assert AdapterRef.from_dict(ref.to_dict()) == ref

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ContractViolationError, match="non-empty"):
            AdapterRef(name="", version=1, kind=AdapterKind.CLIENT)

    def test_rejects_negative_version(self) -> None:
        with pytest.raises(ContractViolationError, match=">= 0"):
            AdapterRef(name="a", version=-1, kind=AdapterKind.CLIENT)

    def test_cluster_adapter_requires_cluster_id(self) -> None:
        with pytest.raises(ContractViolationError, match="cluster_id"):
            AdapterRef(name="a", version=1, kind=AdapterKind.CLUSTER)

    def test_rejects_unknown_kind(self) -> None:
        with pytest.raises(ContractViolationError, match="Invalid AdapterRef"):
            AdapterRef.from_dict({"name": "a", "version": 1, "kind": "sideways"})


class TestEvalResult:
    def test_pass_at_k_keys_serialise_as_strings(self) -> None:
        result = EvalResult(
            adapter=AdapterRef(name="a", version=1, kind=AdapterKind.CLIENT),
            benchmark=BenchmarkName.HUMANEVAL,
            pass_at_k={1: 0.25, 10: 0.5},
        )
        assert set(result.to_dict()["pass_at_k"]) == {"1", "10"}

    def test_round_trips(self) -> None:
        result = EvalResult(
            adapter=AdapterRef(name="a", version=1, kind=AdapterKind.CLIENT),
            benchmark=BenchmarkName.MBPP,
            pass_at_k={1: 0.25},
            num_tasks=5,
        )
        assert EvalResult.from_dict(result.to_dict()) == result

    def test_rejects_out_of_range_rate(self) -> None:
        with pytest.raises(ContractViolationError, match=r"\[0, 1\]"):
            EvalResult(
                adapter=AdapterRef(name="a", version=1, kind=AdapterKind.CLIENT),
                benchmark=BenchmarkName.HUMANEVAL,
                pass_at_k={1: 1.5},
            )

    def test_rejects_k_below_one(self) -> None:
        with pytest.raises(ContractViolationError, match="k >= 1"):
            EvalResult(
                adapter=AdapterRef(name="a", version=1, kind=AdapterKind.CLIENT),
                benchmark=BenchmarkName.HUMANEVAL,
                pass_at_k={0: 0.5},
            )


class TestPartitionManifestContract:
    def test_round_trips(self) -> None:
        manifest = make_manifest()
        assert PartitionManifest.from_dict(manifest.to_dict()).to_dict() == manifest.to_dict()

    def test_rejects_client_count_mismatch(self) -> None:
        with pytest.raises(ContractViolationError, match="num_clients"):
            replace(make_manifest(), num_clients=99)

    def test_rejects_total_files_mismatch(self) -> None:
        with pytest.raises(ContractViolationError, match="total_files"):
            replace(make_manifest(), total_files=1)

    def test_shard_lookup(self) -> None:
        assert make_manifest().shard_for("client-beta").num_files == 6

    def test_shard_rejects_zero_files(self) -> None:
        with pytest.raises(ContractViolationError, match="no files"):
            make_shard(num_files=0)

    def test_carries_contract_version(self) -> None:
        assert make_manifest().contract_version == CONTRACT_VERSION


# ---------------------------------------------------------------------------
# P1 — Edge Layer mock
# ---------------------------------------------------------------------------
class TestMockEdgeInferenceClient:
    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(MockEdgeInferenceClient(), EdgeInferenceClient)

    def test_generates_requested_sample_count(self) -> None:
        client = MockEdgeInferenceClient()
        result = client.generate(GenerationRequest(task_id="t", prompt="def f():\n", num_samples=3))
        assert len(result.completions) == 3
        assert result.is_mock

    def test_is_deterministic(self) -> None:
        request = GenerationRequest(task_id="t", prompt="def f():\n")
        first = MockEdgeInferenceClient().generate(request)
        second = MockEdgeInferenceClient().generate(request)
        assert first.completions == second.completions

    def test_differs_across_tasks(self) -> None:
        client = MockEdgeInferenceClient()
        a = client.generate(GenerationRequest(task_id="a", prompt="p"))
        b = client.generate(GenerationRequest(task_id="b", prompt="p"))
        assert a.completions != b.completions

    def test_samples_differ_within_a_request(self) -> None:
        result = MockEdgeInferenceClient().generate(
            GenerationRequest(task_id="t", prompt="p", num_samples=2)
        )
        assert result.completions[0] != result.completions[1]

    def test_batch_preserves_order(self) -> None:
        requests = [GenerationRequest(task_id=f"t{i}", prompt="p") for i in range(4)]
        results = MockEdgeInferenceClient().generate_batch(requests)
        assert [r.task_id for r in results] == [r.task_id for r in requests]

    def test_injected_failure_raises(self) -> None:
        client = MockEdgeInferenceClient(fail_task_ids=["boom"])
        with pytest.raises(ClaspP5Error, match="injected failure"):
            client.generate(GenerationRequest(task_id="boom", prompt="p"))

    def test_counts_calls(self) -> None:
        client = MockEdgeInferenceClient()
        client.generate(GenerationRequest(task_id="t", prompt="p"))
        assert client.call_count == 1

    def test_rejects_empty_prompt(self) -> None:
        with pytest.raises(ClaspP5Error, match="empty prompt"):
            GenerationRequest(task_id="t", prompt="")


# ---------------------------------------------------------------------------
# P1 — Edge Layer real adapter (Week 3)
# ---------------------------------------------------------------------------
class TestTransformersEdgeInferenceClient:
    """Dependency availability for the real Edge adapter varies by machine —
    this environment has none of torch/transformers/peft/bitsandbytes
    installed, but CI or a reviewer's machine may have some subset (e.g.
    torch alone). These tests must not assume "torch present" implies "every
    dependency present": they use
    ``interfaces.edge_transformers_adapter.missing_dependencies`` — the same
    requirement-detection logic the adapter's own ``_load`` uses — to decide
    which branch is correct, rather than checking one package by hand.
    """

    #: register_transformers_edge_client's defaults (adapters=(), load_in_4bit=True)
    #: are what both tests below actually construct with; the requirement set
    #: must be computed for *that* call shape, not guessed.
    _CONSTRUCTION_KWARGS = {"adapters": (), "load_in_4bit": True}

    def test_module_imports_without_the_heavy_dependencies(self) -> None:
        # If this test file collected at all, the import already succeeded;
        # asserting the symbols exist pins that it stays true after edits.
        assert TransformersEdgeInferenceClient is not None
        assert LoraComposition is not None

    def test_missing_dependency_is_reported_clearly(self) -> None:
        from interfaces.edge_transformers_adapter import missing_dependencies

        missing = missing_dependencies(**self._CONSTRUCTION_KWARGS)
        if not missing:
            pytest.skip(
                "torch, transformers and bitsandbytes are all installed in this environment; "
                "the missing-dependency failure path cannot be exercised"
            )
        with pytest.raises(DependencyError, match="pip install"):
            TransformersEdgeInferenceClient("deepseek-ai/deepseek-coder-6.7b-base")

    def test_registering_makes_the_edge_backend_reachable(self, evaluation_config) -> None:
        """Week-3 Mon/Tue: once P1 registers a factory, backend.kind='edge' stops
        being rejected outright — the seam is live, even before real weights exist.

        Registration mutates module-level state in evaluation.registry, so the
        factory is saved and restored around the test to avoid leaking into
        other tests regardless of collection order.
        """
        import evaluation.registry as registry_module
        from evaluation.registry import build_inference_client
        from interfaces.edge_transformers_adapter import missing_dependencies

        previous_factory = registry_module._EDGE_CLIENT_FACTORY
        try:
            register_transformers_edge_client("deepseek-ai/deepseek-coder-6.7b-base")
            config = replace(evaluation_config, backend=replace(evaluation_config.backend, kind="edge"))

            # Deciding the branch on "is torch importable?" alone is wrong: a
            # machine can have torch without transformers (or without
            # bitsandbytes, needed here because register_transformers_edge_client
            # defaults load_in_4bit=True), and construction still raises
            # DependencyError in that case. Check every dependency this
            # specific call actually needs.
            missing = missing_dependencies(**self._CONSTRUCTION_KWARGS)
            if missing:
                with pytest.raises(DependencyError, match="pip install"):
                    build_inference_client(config)
            else:  # pragma: no cover - only when every dependency is actually installed
                client = build_inference_client(config)
                assert isinstance(client, TransformersEdgeInferenceClient)
        finally:
            registry_module._EDGE_CLIENT_FACTORY = previous_factory


# ---------------------------------------------------------------------------
# P2 — Cluster Layer mock
# ---------------------------------------------------------------------------
class TestMockClusterClient:
    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(MockClusterClient(), ClusterPlanConsumer)

    def test_accepts_a_valid_plan(self) -> None:
        acceptance = MockClusterClient(expected_client_count=2).register_partition_plan(make_manifest())
        assert acceptance.accepted
        assert not acceptance.errors
        assert not acceptance.warnings

    def test_warns_on_client_count_mismatch(self) -> None:
        acceptance = MockClusterClient(expected_client_count=3).register_partition_plan(make_manifest())
        assert acceptance.accepted
        assert any("widen the simulation" in w for w in acceptance.warnings)

    def test_rejects_a_single_client_federation(self) -> None:
        manifest = make_manifest([make_shard("client-solo", 4)])
        acceptance = MockClusterClient().register_partition_plan(manifest)
        assert not acceptance.accepted
        assert any(">= 2 clients" in e for e in acceptance.errors)

    def test_rejects_too_many_clients(self) -> None:
        manifest = make_manifest([make_shard(f"client-{i}", 2) for i in range(5)])
        acceptance = MockClusterClient(max_client_count=3).register_partition_plan(manifest)
        assert not acceptance.accepted
        assert any("caps at" in e for e in acceptance.errors)

    def test_records_the_accepted_manifest(self) -> None:
        client = MockClusterClient(expected_client_count=2)
        client.register_partition_plan(make_manifest())
        assert client.last_manifest is not None

    def test_does_not_record_a_rejected_manifest(self) -> None:
        client = MockClusterClient()
        client.register_partition_plan(make_manifest([make_shard("client-solo", 1)]))
        assert client.last_manifest is None

    def test_summary_is_human_readable(self) -> None:
        acceptance: PlanAcceptance = MockClusterClient(2).register_partition_plan(make_manifest())
        assert "accepted" in acceptance.summary()


# ---------------------------------------------------------------------------
# P4 — Registry mock
# ---------------------------------------------------------------------------
class TestMockRegistryClient:
    @pytest.fixture
    def registry(self) -> MockRegistryClient:
        client = MockRegistryClient()
        for version in (1, 2, 3):
            client.add_snapshot(
                AdapterRef(name="flask", version=version, kind=AdapterKind.CLUSTER, cluster_id="flask"),
                round_number=version,
                artifact_path=f"registry/flask/v{version}/adapter.safetensors",
            )
        return client

    def test_satisfies_the_protocol(self, registry: MockRegistryClient) -> None:
        assert isinstance(registry, RegistryReadClient)

    def test_lists_versions_oldest_first(self, registry: MockRegistryClient) -> None:
        versions = registry.list_versions("flask", AdapterKind.CLUSTER)
        assert [s.adapter.version for s in versions] == [1, 2, 3]

    def test_latest_returns_newest(self, registry: MockRegistryClient) -> None:
        assert registry.latest("flask", AdapterKind.CLUSTER).adapter.version == 3

    def test_latest_raises_for_unknown_adapter(self, registry: MockRegistryClient) -> None:
        with pytest.raises(SnapshotNotFoundError):
            registry.latest("nope", AdapterKind.CLUSTER)

    def test_load_metadata_by_exact_ref(self, registry: MockRegistryClient) -> None:
        ref = AdapterRef(name="flask", version=2, kind=AdapterKind.CLUSTER, cluster_id="flask")
        assert registry.load_metadata(ref).round_number == 2

    def test_load_metadata_rejects_unknown_version(self, registry: MockRegistryClient) -> None:
        ref = AdapterRef(name="flask", version=99, kind=AdapterKind.CLUSTER, cluster_id="flask")
        with pytest.raises(SnapshotNotFoundError):
            registry.load_metadata(ref)

    def test_kind_is_part_of_the_lookup_key(self, registry: MockRegistryClient) -> None:
        assert registry.list_versions("flask", AdapterKind.CLIENT) == []

    def test_from_directory_reads_sidecars(self, tmp_path: Path) -> None:
        from utils.io_utils import write_json

        ref = AdapterRef(name="click", version=1, kind=AdapterKind.CLUSTER, cluster_id="click")
        snapshot = SnapshotMetadata(
            adapter=ref,
            round_number=1,
            created_at=utc_timestamp(),
            artifact_path="registry/click/v1/adapter.safetensors",
        )
        write_json(tmp_path / "click_v1.json", snapshot.to_dict())

        client = MockRegistryClient.from_directory(tmp_path)
        assert len(client) == 1
        assert client.latest("click", AdapterKind.CLUSTER).adapter == ref

    def test_from_directory_skips_unrelated_json(self, tmp_path: Path) -> None:
        from utils.io_utils import write_json

        write_json(tmp_path / "unrelated.json", {"hello": "world"})
        assert len(MockRegistryClient.from_directory(tmp_path)) == 0

    def test_from_directory_rejects_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(ClaspP5Error, match="not found"):
            MockRegistryClient.from_directory(tmp_path / "absent")


# ---------------------------------------------------------------------------
# P3 — Security placeholder
# ---------------------------------------------------------------------------
class TestPrivacyAccountant:
    def test_null_accountant_satisfies_the_protocol(self) -> None:
        assert isinstance(NullPrivacyAccountant(), PrivacyAccountant)

    def test_null_accountant_reports_dp_off(self) -> None:
        budget = NullPrivacyAccountant().current_budget()
        assert not budget.enabled
        assert budget.epsilon is None
        assert budget.label() == "DP off (baseline)"

    def test_enabled_budget_labels_epsilon(self) -> None:
        budget = PrivacyBudget(epsilon=3.14159, delta=1e-5, enabled=True)
        assert "ε=3.14" in budget.label()

    def test_enabled_budget_requires_epsilon(self) -> None:
        with pytest.raises(ClaspP5Error, match="no epsilon"):
            PrivacyBudget(enabled=True)

    def test_rejects_negative_epsilon(self) -> None:
        with pytest.raises(ClaspP5Error, match="non-negative"):
            PrivacyBudget(epsilon=-1.0, enabled=True)


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------
class TestContractValidation:
    def test_jsonschema_is_available_in_this_environment(self) -> None:
        assert jsonschema_available(), "install jsonschema so contract checks are meaningful"

    @pytest.mark.parametrize("contract", sorted(SCHEMA_REGISTRY))
    def test_every_registered_schema_loads(self, contract: str) -> None:
        schema = load_schema(contract)
        assert schema["$schema"].startswith("https://json-schema.org/")
        assert "properties" in schema

    def test_valid_manifest_passes(self) -> None:
        report = validate_document("partition_manifest", make_manifest().to_dict())
        assert report.ok, report.errors
        assert "json-schema structural validation" in report.checks_run

    def test_missing_required_field_fails(self) -> None:
        payload = make_manifest().to_dict()
        del payload["seed"]
        assert not validate_document("partition_manifest", payload).ok

    def test_bad_sha_pattern_fails(self) -> None:
        payload = make_manifest().to_dict()
        payload["corpus_sha256"] = "not-a-sha"
        assert not validate_document("partition_manifest", payload).ok

    def test_unknown_extra_field_fails(self) -> None:
        payload = make_manifest().to_dict()
        payload["surprise"] = True
        assert not validate_document("partition_manifest", payload).ok

    def test_bad_client_id_pattern_fails(self) -> None:
        payload = make_manifest().to_dict()
        payload["shards"][0]["client_id"] = "Client Alpha!"
        assert not validate_document("partition_manifest", payload).ok

    def test_semantic_layer_catches_count_drift(self) -> None:
        payload = make_manifest().to_dict()
        payload["total_files"] = 999
        report = validate_document("partition_manifest", payload)
        assert not report.ok
        assert any("semantic" in error for error in report.errors)

    def test_valid_eval_result_passes(self) -> None:
        result = EvalResult(
            adapter=AdapterRef(name="a", version=1, kind=AdapterKind.CLIENT),
            benchmark=BenchmarkName.HUMANEVAL,
            pass_at_k={1: 0.3},
            num_tasks=10,
            created_at=utc_timestamp(),
        )
        assert validate_document("eval_result", result.to_dict()).ok

    def test_valid_snapshot_metadata_passes(self) -> None:
        snapshot = SnapshotMetadata(
            adapter=AdapterRef(name="flask", version=1, kind=AdapterKind.CLUSTER, cluster_id="flask"),
            round_number=1,
            created_at=utc_timestamp(),
            artifact_path="registry/flask/v1/adapter.safetensors",
            artifact_sha256="c" * 64,
        )
        assert validate_document("snapshot_metadata", snapshot.to_dict()).ok

    def test_unknown_contract_raises(self) -> None:
        with pytest.raises(ClaspP5Error, match="Unknown contract"):
            load_schema("not_a_contract")

    def test_validate_file_round_trip(self, tmp_path: Path) -> None:
        from utils.io_utils import write_json

        path = write_json(tmp_path / "manifest.json", make_manifest().to_dict())
        assert validate_file("partition_manifest", path).ok

    def test_raise_for_status_raises_on_failure(self) -> None:
        payload = make_manifest().to_dict()
        del payload["seed"]
        with pytest.raises(ContractViolationError):
            validate_document("partition_manifest", payload).raise_for_status()

    def test_report_summary_is_readable(self) -> None:
        assert "PASS" in validate_document("partition_manifest", make_manifest().to_dict()).summary()

    def test_repository_manifest_satisfies_the_contract(self) -> None:
        """The real, committed manifest must validate — a live regression guard."""
        from partitions.partitioner import load_partition_config

        manifest_path = Path(load_partition_config().output.manifest_path)
        if not manifest_path.is_file():
            pytest.skip("run scripts/build_partitions.py first")
        report = validate_file("partition_manifest", manifest_path)
        assert report.ok, report.errors
