"""P1 (Edge Layer) boundary — inference protocol plus a deterministic mock.

**P5 does not implement the Edge Layer.** P1 (Adithyaa) owns 4-bit DeepSeek
loading, LoRA training and the Dynamic Merger. P5 only *calls* it, and only
from Week 3 onward ("wire HumanEval runner to call the merged model").

Weeks 1-2 therefore need exactly two things:

1. :class:`EdgeInferenceClient` — the ``Protocol`` P1 will satisfy, so the
   harness is written against a stable signature now.
2. :class:`MockEdgeInferenceClient` — a deterministic stand-in that lets the
   Week-1 Friday harness dry-run exercise every code path with no GPU,
   no model weights and no network.

The mock never claims to produce correct code. It produces *syntactically
plausible, deterministic* completions whose only job is to prove the harness
plumbing (task loading, prompt assembly, batching, artefact writing) works.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from interfaces.contracts import AdapterRef
from utils.errors import ClaspP5Error
from utils.logging_utils import get_logger

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class GenerationRequest:
    """One completion request handed to the Edge Layer."""

    task_id: str
    prompt: str
    max_new_tokens: int = 384
    temperature: float = 0.2
    stop_sequences: tuple[str, ...] = ()
    num_samples: int = 1

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ClaspP5Error(f"GenerationRequest for {self.task_id} has an empty prompt")
        if self.num_samples < 1:
            raise ClaspP5Error("num_samples must be >= 1")


@dataclass(frozen=True)
class GenerationResult:
    """Completions returned for a single task.

    ``completions`` has length ``GenerationRequest.num_samples``; Pass@k
    scoring (Week 3) consumes exactly this shape.
    """

    task_id: str
    completions: list[str]
    model_id: str
    adapters: list[AdapterRef] = field(default_factory=list)
    latency_ms: float | None = None
    is_mock: bool = False


@runtime_checkable
class EdgeInferenceClient(Protocol):
    """Contract P1's Edge Layer must satisfy for P5's harness to drive it.

    Implementations are expected to be stateful (a loaded model) and are used
    as context managers so the harness can release GPU memory deterministically.
    """

    @property
    def model_id(self) -> str:
        """Identifier of the base model, e.g. ``deepseek-ai/deepseek-coder-6.7b-base``."""
        ...

    def loaded_adapters(self) -> Sequence[AdapterRef]:
        """Adapters currently composed into the served weights."""
        ...

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Produce completions for a single prompt."""
        ...

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        """Produce completions for several prompts.

        Implementations may batch on the accelerator; the harness only relies
        on the returned order matching the request order.
        """
        ...


class MockEdgeInferenceClient:
    """Deterministic, offline stand-in for P1's Edge Layer.

    Determinism comes from hashing ``(task_id, prompt, sample_index)``, so a
    dry-run produces byte-identical artefacts across machines — which is what
    makes the Week-1 Friday dry-run a usable regression check rather than a
    one-off smoke test.

    Args:
        model_id: Value reported through :attr:`model_id`.
        adapters: Adapter refs to report as composed, for logging realism.
        fail_task_ids: Task ids for which :meth:`generate` raises. Used by the
            test suite to exercise the harness's per-task error handling.
    """

    def __init__(
        self,
        model_id: str = "mock/deepseek-coder-6.7b-base",
        adapters: Sequence[AdapterRef] | None = None,
        fail_task_ids: Sequence[str] = (),
    ) -> None:
        self._model_id = model_id
        self._adapters = list(adapters or [])
        self._fail_task_ids = set(fail_task_ids)
        self._call_count = 0

    # --- EdgeInferenceClient -------------------------------------------
    @property
    def model_id(self) -> str:
        return self._model_id

    def loaded_adapters(self) -> Sequence[AdapterRef]:
        return tuple(self._adapters)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self._call_count += 1
        if request.task_id in self._fail_task_ids:
            raise ClaspP5Error(f"MockEdgeInferenceClient: injected failure for {request.task_id}")

        completions = [
            self._synthesise(request, sample_index) for sample_index in range(request.num_samples)
        ]
        _LOG.debug("mock generate task_id=%s samples=%d", request.task_id, len(completions))
        return GenerationResult(
            task_id=request.task_id,
            completions=completions,
            model_id=self._model_id,
            adapters=list(self._adapters),
            latency_ms=0.0,
            is_mock=True,
        )

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        return [self.generate(request) for request in requests]

    # --- introspection for tests ----------------------------------------
    @property
    def call_count(self) -> int:
        """Number of :meth:`generate` invocations, including failed ones."""
        return self._call_count

    # --- internals -------------------------------------------------------
    @staticmethod
    def _synthesise(request: GenerationRequest, sample_index: int) -> str:
        """Return a deterministic placeholder body for ``request``."""
        seed = hashlib.sha256(
            f"{request.task_id}|{request.prompt}|{sample_index}".encode("utf-8")
        ).hexdigest()[:8]
        return (
            f"    # MOCK COMPLETION (seed={seed}, sample={sample_index})\n"
            f"    # Generated by MockEdgeInferenceClient; P1 replaces this from Week 3.\n"
            "    raise NotImplementedError('mock completion')\n"
        )
