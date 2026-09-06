"""Benchmark adapter registry and inference-backend factory.

The composition root for the evaluation stack: it is the one place that knows
which concrete adapter serves which benchmark, and which client the harness
talks to. Swapping the mock Edge client for P1's real one in Week 3 is a
change to :func:`build_inference_client` alone.
"""

from __future__ import annotations

from typing import Callable, Final

from evaluation.base import BenchmarkAdapter
from evaluation.humaneval.adapter import HumanEvalAdapter
from evaluation.mbpp.adapter import MbppAdapter
from evaluation.models import BackendConfig, EvaluationConfig
from interfaces.contracts import BenchmarkName
from interfaces.edge_client import EdgeInferenceClient, MockEdgeInferenceClient
from utils.errors import EvaluationError
from utils.logging_utils import get_logger

_LOG = get_logger(__name__)

#: Benchmark -> adapter class.
ADAPTERS: Final[dict[BenchmarkName, type[BenchmarkAdapter]]] = {
    BenchmarkName.HUMANEVAL: HumanEvalAdapter,
    BenchmarkName.MBPP: MbppAdapter,
}


def build_adapter(name: BenchmarkName, config: EvaluationConfig) -> BenchmarkAdapter:
    """Instantiate the adapter registered for ``name``."""
    try:
        adapter_cls = ADAPTERS[name]
    except KeyError as exc:  # pragma: no cover - enum-guarded
        raise EvaluationError(f"No adapter registered for benchmark '{name}'") from exc
    return adapter_cls(config.benchmark(name))


def build_adapters(config: EvaluationConfig) -> list[BenchmarkAdapter]:
    """Instantiate an adapter for every benchmark enabled in config."""
    adapters = [build_adapter(name, config) for name in config.enabled_benchmarks()]
    if not adapters:
        raise EvaluationError(
            "No benchmarks are enabled in configs/evaluation.yaml; nothing to evaluate"
        )
    return adapters


#: Optional hook for P1 to register the real Edge client without editing P5.
#: Week 3 sets this at the composition root; until then it stays ``None``.
_EDGE_CLIENT_FACTORY: Callable[[BackendConfig], EdgeInferenceClient] | None = None


def register_edge_client_factory(
    factory: Callable[[BackendConfig], EdgeInferenceClient],
) -> None:
    """Register the factory that builds P1's real Edge Layer client.

    Keeps the Week-3 integration additive: P1 registers a factory and flips
    ``backend.kind`` to ``edge``; no P5 module needs to import P1's package.
    """
    global _EDGE_CLIENT_FACTORY
    _EDGE_CLIENT_FACTORY = factory
    _LOG.info("Registered Edge Layer client factory: %s", getattr(factory, "__name__", factory))


def build_inference_client(config: EvaluationConfig) -> EdgeInferenceClient:
    """Build the inference client named by ``backend.kind``.

    Raises:
        EvaluationError: If the real Edge backend is requested before P1 has
            registered a factory. Failing loudly here beats silently
            substituting the mock and publishing meaningless Pass@k numbers.
    """
    backend = config.backend

    if backend.kind == "mock":
        _LOG.info("Using MockEdgeInferenceClient (offline, deterministic)")
        return MockEdgeInferenceClient(model_id=backend.model_id)

    if backend.kind == "edge":
        if _EDGE_CLIENT_FACTORY is None:
            raise EvaluationError(
                "backend.kind='edge' requires P1's Edge Layer, which is not integrated until "
                "Week 3. Register a factory via evaluation.registry.register_edge_client_factory, "
                "or set backend.kind='mock' in configs/evaluation.yaml."
            )
        return _EDGE_CLIENT_FACTORY(backend)

    raise EvaluationError(f"Unsupported backend.kind '{backend.kind}'")  # pragma: no cover
