"""P5 <-> State Registry (P4) over the registry's real HTTP API and frozen contracts v1.0.

:mod:`interfaces.contracts` is P5's internal mirror; it predates the frozen
``contracts`` package and its ``EvalResult`` is P5's *results-store* record
(one benchmark, flat ``pass_at_k``), not the wire type. The registry speaks
``contracts`` v1.0. This module is the only place the two meet, in both
directions:

Outbound — what P5 sends (seam C2, ``POST /adapters/{name}/promote``)
    :func:`guard_metrics`, :func:`in_project_wire`, :func:`eval_result_wire`
    and :func:`promote_body` build exactly the JSON
    ``registry.app._eval_result_from`` parses — the same field set
    ``edge.promote.build_eval_result`` sends. P5's HumanEval/MBPP pass@k
    becomes ``EvalResult.guard``; the in-project metric becomes
    ``EvalResult.in_project``.

Inbound — what P5 reads (seam C1 metadata)
    :class:`HttpRegistryReadClient` implements
    :class:`~interfaces.registry_client.RegistryReadClient` against
    ``GET /adapters/{name}/versions[/{v}]`` and ``GET /adapters/{name}/active``,
    turning the registry's ``AdapterMetadata`` documents into P5's
    :class:`~interfaces.contracts.SnapshotMetadata`.

P5 does not move adapter bytes: downloading, sha256 verification and PEFT
materialization belong to ``edge.registry_client`` (B4). P5 does not decide
promotion either: the D5 rule is ``registry.promotion.decide``. This module
only gets P5's numbers onto the wire in the shape that rule reads.

No third-party dependency: HTTP goes through :mod:`urllib` unless a
``requests``-style session (``.get(url)`` returning ``status_code`` / ``.json()``)
is injected — FastAPI's ``TestClient`` is one, which is how the tests drive
the real registry app in-process.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable, Mapping, Protocol, Sequence

from evaluation.interfaces.contracts import (
    AdapterKind,
    AdapterRef,
    BenchmarkName,
    EvalResult,
    InProjectMetrics,
    SnapshotMetadata,
)
from evaluation.interfaces.registry_client import SnapshotNotFoundError
from evaluation.utils.errors import ClaspP5Error, ContractViolationError

#: The contracts version of the payloads this module writes. The edge lane sends the
#: same v1.0 shape; contracts v1.1 (additive) parses it unchanged.
WIRE_CONTRACTS_VERSION = "1.0.0"

#: The benchmark the D5 guard reads (``registry.promotion._guard_holds``).
GUARD_BENCHMARK = BenchmarkName.HUMANEVAL.value


# ---------------------------------------------------------------------------
# Outbound: P5 metrics -> contracts v1.0 JSON
# ---------------------------------------------------------------------------
def guard_metrics(results: Iterable[EvalResult]) -> list[dict[str, Any]]:
    """``contracts.GuardMetrics`` dicts from P5's per-benchmark results.

    One entry per benchmark, ``pass_at_k`` keyed by ``str(k)`` (JSON keys are
    strings; the registry casts them back with ``int(k)``). Two results for the
    same benchmark are refused: the guard is one run's number, and silently
    picking one of two would hide which.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in results:
        name = result.benchmark.value
        if name in seen:
            raise ContractViolationError(
                f"Two {name} results passed to guard_metrics; pass exactly one run per benchmark"
            )
        if not result.pass_at_k:
            raise ContractViolationError(f"{name} result has no pass@k values; nothing to guard with")
        seen.add(name)
        out.append(
            {
                "benchmark": name,
                "pass_at_k": {str(k): float(v) for k, v in sorted(result.pass_at_k.items())},
            }
        )
    return out


def in_project_wire(metrics: InProjectMetrics, *, perplexity: float | None = None) -> dict[str, Any]:
    """``contracts.InProjectMetrics`` dict — perplexity is mandatory on the wire.

    contracts v1.0 types ``perplexity`` as a float. P5 has no logits, so the
    number comes from P1's held-out evaluation (``edge.train_client.evaluate``)
    and is passed here, or already sits on ``metrics``. With neither, this
    raises rather than inventing one.
    """
    ppl = perplexity if perplexity is not None else metrics.perplexity
    if ppl is None:
        raise ContractViolationError(
            "contracts v1.0 InProjectMetrics requires perplexity; P5 does not measure it — "
            "pass P1's held-out perplexity (edge.train_client.evaluate) at the seam"
        )
    ppl = float(ppl)
    if math.isnan(ppl) or ppl <= 0.0:
        raise ContractViolationError(f"perplexity must be a positive number, got {ppl}")
    if metrics.n_examples < 1:
        raise ContractViolationError(
            "in-project metrics with n_examples=0 are not a measurement; refusing to send them"
        )
    return {
        "edit_similarity": float(metrics.edit_similarity),
        "exact_match": float(metrics.exact_match),
        "perplexity": ppl,
        "n_examples": int(metrics.n_examples),
    }


def eval_result_wire(
    adapter: AdapterRef,
    in_project: Mapping[str, Any],
    *,
    guard: Sequence[Mapping[str, Any]] = (),
    baseline_in_project: Mapping[str, Any] | None = None,
    baseline_noise_band: float,
    seed: int = 0,
) -> dict[str, Any]:
    """The ``contracts.EvalResult`` JSON body (the ``eval`` field of a promote call).

    Same keys as ``edge.promote.build_eval_result``. ``in_project`` /
    ``baseline_in_project`` are :func:`in_project_wire` dicts; ``guard`` is
    :func:`guard_metrics` output. ``baseline_noise_band`` has no default: every
    result that feeds promotion must carry the band measured from repeated
    baseline evaluations (D5), so a caller cannot send one by omission.
    """
    if baseline_noise_band < 0.0 or math.isnan(baseline_noise_band):
        raise ContractViolationError(f"baseline_noise_band must be >= 0, got {baseline_noise_band}")
    return {
        "adapter": {
            "name": adapter.name,
            "version": adapter.version,
            "kind": adapter.kind.value,
            "cluster_id": adapter.cluster_id,
        },
        "in_project": dict(in_project),
        "guard": [dict(g) for g in guard],
        "baseline_in_project": dict(baseline_in_project) if baseline_in_project is not None else None,
        "baseline_noise_band": float(baseline_noise_band),
        "seed": int(seed),
    }


def promote_body(
    eval_result: Mapping[str, Any], baseline_guard: Sequence[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    """``POST /adapters/{name}/promote`` body: ``{"eval": ..., "baseline_guard": [...]}``.

    ``baseline_guard`` is the previous active version's guard metrics — an
    API-boundary extension the registry documents next to the endpoint.
    """
    return {"eval": dict(eval_result), "baseline_guard": [dict(g) for g in baseline_guard]}


def to_contracts_eval_result(wire: Mapping[str, Any]):
    """Parse a wire dict into the real ``contracts.EvalResult``.

    Uses ``contracts.EvalResult.from_json`` — exactly what ``registry.app``
    calls on ``POST /promote`` (contracts v1.1, which accepts v1.0 payloads
    unchanged) — so a payload that parses here parses there. Needs the
    ``contracts`` package, which ``clasp-evaluation`` depends on.
    """
    import contracts as c

    return c.EvalResult.from_json(dict(wire))


# ---------------------------------------------------------------------------
# Inbound: registry AdapterMetadata -> P5 SnapshotMetadata
# ---------------------------------------------------------------------------
def snapshot_from_registry(meta: Mapping[str, Any]) -> SnapshotMetadata:
    """Map one registry ``AdapterMetadata`` document onto P5's ``SnapshotMetadata``.

    ``artifact_path`` is the registry-relative download route for the blob;
    ``artifact_sha256`` is the digest the registry recorded, which is what
    ``edge.registry_client`` verifies the download against.
    """
    try:
        ref = meta["ref"]
        adapter = AdapterRef(
            name=str(ref["name"]),
            version=int(ref["version"]),
            kind=AdapterKind(ref["kind"]),
            cluster_id=ref.get("cluster_id"),
        )
        return SnapshotMetadata(
            adapter=adapter,
            round_number=int(meta["round"]) if meta.get("round") is not None else 0,
            created_at=str(meta["created_at"]),
            artifact_path=f"/adapters/{adapter.name}/versions/{adapter.version}/file",
            artifact_sha256=str(meta["sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractViolationError(f"Not a registry AdapterMetadata document: {exc}") from exc


class _Response(Protocol):
    status_code: int

    def json(self) -> Any: ...


class _Session(Protocol):
    def get(self, url: str) -> _Response: ...


class _UrllibResponse:
    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        return json.loads(self._body.decode("utf-8"))


class _UrllibSession:
    """Minimal stdlib transport so P5 needs no HTTP library."""

    def __init__(self, timeout: float) -> None:
        self._timeout = timeout

    def get(self, url: str) -> _UrllibResponse:
        try:
            with urllib.request.urlopen(url, timeout=self._timeout) as resp:  # noqa: S310 - caller-supplied registry URL
                return _UrllibResponse(resp.status, resp.read())
        except urllib.error.HTTPError as exc:
            return _UrllibResponse(exc.code, exc.read() or b"{}")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ClaspP5Error(f"State Registry unreachable at {url}: {exc}") from exc


class HttpRegistryReadClient:
    """:class:`~interfaces.registry_client.RegistryReadClient` over the real registry API.

    ``base_url`` is e.g. ``http://localhost:8004`` (the compose port). Pass
    ``session`` to reuse a ``requests.Session`` or a FastAPI ``TestClient``
    (with ``base_url=""``).
    """

    def __init__(self, base_url: str, *, session: _Session | None = None, timeout: float = 10.0) -> None:
        self._base = base_url.rstrip("/")
        self._session: _Session = session if session is not None else _UrllibSession(timeout)

    def _get(self, path: str) -> Any:
        url = f"{self._base}{path}"
        resp = self._session.get(url)
        if resp.status_code == 404:
            raise SnapshotNotFoundError(f"registry 404 for {path}")
        if resp.status_code != 200:
            raise ClaspP5Error(f"registry GET {path} returned HTTP {resp.status_code}")
        return resp.json()

    @staticmethod
    def _q(name: str) -> str:
        return urllib.parse.quote(name, safe="")

    # --- RegistryReadClient ------------------------------------------------
    def list_versions(self, name: str, kind: AdapterKind) -> Sequence[SnapshotMetadata]:
        """Every version of ``name`` whose kind is ``kind``, oldest first."""
        try:
            payload = self._get(f"/adapters/{self._q(name)}/versions")
        except SnapshotNotFoundError:
            return []
        snapshots = [snapshot_from_registry(v) for v in payload.get("versions", [])]
        return sorted((s for s in snapshots if s.adapter.kind is kind), key=lambda s: s.adapter.version)

    def latest(self, name: str, kind: AdapterKind) -> SnapshotMetadata:
        """Highest version of ``name`` (not necessarily the active one — see :meth:`active`)."""
        versions = self.list_versions(name, kind)
        if not versions:
            raise SnapshotNotFoundError(f"No snapshots for {kind.value}/{name}")
        return versions[-1]

    def load_metadata(self, ref: AdapterRef) -> SnapshotMetadata:
        snapshot = snapshot_from_registry(
            self._get(f"/adapters/{self._q(ref.name)}/versions/{int(ref.version)}")
        )
        if snapshot.adapter.kind is not ref.kind:
            raise ContractViolationError(
                f"{ref.name} v{ref.version} is a {snapshot.adapter.kind.value} adapter, not {ref.kind.value}"
            )
        return snapshot

    # --- beyond the protocol ----------------------------------------------
    def active(self, name: str) -> SnapshotMetadata:
        """The version the registry currently serves — the only one D5's promote judges."""
        return snapshot_from_registry(self._get(f"/adapters/{self._q(name)}/active"))
