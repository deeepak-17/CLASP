"""Real P1 (Edge Layer) client — Week 3, "wire the runner to the merged model".

**P5 still does not implement the Edge Layer.** P1 (Adithyaa) owns 4-bit
DeepSeek-Coder-6.7B loading, LoRA training and the Dynamic Merger
(``y = W_base(x) + beta * B_u(A_u(x))``, Phase-II Week-4 Monday). What P5
owns, and what this module is, is the *calling* side of the
:class:`~interfaces.edge_client.EdgeInferenceClient` protocol: a real,
runnable adapter that loads a HF causal-LM (optionally with PEFT/LoRA
adapters composed in) and drives ``generate()`` against it, so that the day
P1 publishes a merged checkpoint, pointing this class at it is a config
change, not a code change.

Why this is a separate module from :mod:`interfaces.edge_client`
------------------------------------------------------------------
``interfaces/edge_client.py`` has zero third-party dependencies by design
(see its module docstring) so every Week-1/2 test runs on a laptop with no
GPU. ``torch``/``transformers``/``peft`` are heavy, optional, and declared in
:mod:`evaluation.dependencies` as *not required until Week 3*. Importing them
unconditionally at module load time would make this file — and therefore
anything that imports :mod:`evaluation.registry` — fail on exactly the
machines the Week-1 dry run promises to run on. So the imports are deferred
into ``__init__``, and constructing this class without the dependencies
installed raises a clear, actionable error instead of an opaque
``ModuleNotFoundError`` from three frames down.

Status in this repository (Week 3)
-----------------------------------
This adapter is implemented and unit-testable (its request/response shaping,
prompt handling, and the missing-dependency error path all run today), but
it is **not exercised against real DeepSeek-Coder-6.7B weights** in this
environment: that needs a GPU (or a slow CPU 4-bit path), a Hugging Face
download of a multi-gigabyte checkpoint, and — for the merged path
specifically — P1's trained LoRA adapters, none of which are available here.
Every Pass@k number this repository reports therefore comes from
:class:`~interfaces.edge_client.MockEdgeInferenceClient`, labelled
``DEMO/TEST`` throughout. Swapping the backend once weights/adapters exist is
exactly the two-line change demonstrated in
:func:`register_transformers_edge_client`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from interfaces.contracts import AdapterRef
from interfaces.edge_client import GenerationRequest, GenerationResult
from utils.errors import DependencyError
from utils.logging_utils import get_logger
from utils.timing import Stopwatch

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

_LOG = get_logger(__name__)

#: Packages this adapter needs, matching evaluation.dependencies' declarations.
_REQUIRED_PACKAGES = ("torch", "transformers", "peft")


@dataclass(frozen=True)
class LoraComposition:
    """One LoRA adapter to compose onto the base model, and where to find it.

    Mirrors :class:`~interfaces.contracts.AdapterRef` plus the on-disk path
    P4's Registry would hand back from a ``load`` call. P5 does not resolve
    Registry paths itself in Week 3 (that lands Week 7, "eval auto-trigger
    reading the latest Registry snapshot") — a caller passes the path directly.
    """

    ref: AdapterRef
    local_path: str


class TransformersEdgeInferenceClient:
    """Real :class:`~interfaces.edge_client.EdgeInferenceClient` over HF Transformers + PEFT.

    Loads ``model_id`` once (optionally in 4-bit via ``bitsandbytes``,
    matching P1's Week-1 baseline), composes any :class:`LoraComposition`
    adapters onto it via ``peft.PeftModel``, and serves
    :meth:`generate`/:meth:`generate_batch` by calling the model's own
    ``generate()``.

    Args:
        model_id: HF Hub id or local path of the base causal-LM, e.g.
            ``"deepseek-ai/deepseek-coder-6.7b-base"``.
        adapters: LoRA adapters to compose on top of the base model, applied
            in order. Empty means "base model only" — useful for measuring
            the Week-1 baseline arm before any adapter exists.
        load_in_4bit: Match P1's Week-1 "load DeepSeek-Coder-6.7B in 4-bit".
        device_map: Passed through to ``from_pretrained``. ``"auto"`` lets
            ``accelerate`` place layers on whatever accelerator is present.
        dtype: Compute dtype for non-quantised weights.

    Raises:
        DependencyError: If ``torch``, ``transformers`` or ``peft`` (only
            needed when ``adapters`` is non-empty) is not installed, or if
            model/adapter loading itself fails (missing weights, no GPU for a
            configuration that requires one, etc). The message names the
            exact missing piece so it can be reported rather than silently
            substituted with the mock client.
    """

    def __init__(
        self,
        model_id: str,
        *,
        adapters: Sequence[LoraComposition] = (),
        load_in_4bit: bool = True,
        device_map: str = "auto",
        dtype: str = "bfloat16",
    ) -> None:
        self._model_id = model_id
        self._adapter_refs = [composition.ref for composition in adapters]
        self._model, self._tokenizer = self._load(
            model_id, adapters=adapters, load_in_4bit=load_in_4bit, device_map=device_map, dtype=dtype
        )

    # --- EdgeInferenceClient -------------------------------------------
    @property
    def model_id(self) -> str:
        return self._model_id

    def loaded_adapters(self) -> Sequence[AdapterRef]:
        return tuple(self._adapter_refs)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        results: list[GenerationResult] = []
        for request in requests:
            with Stopwatch(request.task_id) as watch:
                completions = self._generate_one(request)
            results.append(
                GenerationResult(
                    task_id=request.task_id,
                    completions=completions,
                    model_id=self._model_id,
                    adapters=list(self._adapter_refs),
                    latency_ms=watch.elapsed_seconds * 1000.0,
                    is_mock=False,
                )
            )
        return results

    # --- internals -------------------------------------------------------
    def _generate_one(self, request: GenerationRequest) -> list[str]:
        import torch

        inputs = self._tokenizer(request.prompt, return_tensors="pt").to(self._model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=request.temperature > 0,
                temperature=max(request.temperature, 1e-5),
                num_return_sequences=request.num_samples,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        completions = []
        for row in output:
            text = self._tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
            completions.append(text)
        return completions

    @staticmethod
    def _load(
        model_id: str,
        *,
        adapters: Sequence[LoraComposition],
        load_in_4bit: bool,
        device_map: str,
        dtype: str,
    ) -> tuple["PreTrainedModel", "PreTrainedTokenizerBase"]:
        missing = [pkg for pkg in ("torch", "transformers") if not _is_importable(pkg)]
        if adapters and not _is_importable("peft"):
            missing.append("peft")
        if missing:
            raise DependencyError(
                f"TransformersEdgeInferenceClient requires {', '.join(missing)}, which "
                f"{'is' if len(missing) == 1 else 'are'} not installed in this environment. "
                f"Install with `pip install {' '.join(missing)}` (and a CUDA-matched `torch` build "
                f"for GPU inference — see https://pytorch.org/get-started/locally/). "
                f"Until then, set backend.kind: mock in configs/evaluation.yaml."
            )

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - covered by the missing-package branch above
            raise DependencyError(f"Failed to import the Edge Layer's model runtime: {exc}") from exc

        quantisation_kwargs: dict = {}
        if load_in_4bit:
            if not _is_importable("bitsandbytes"):
                raise DependencyError(
                    "load_in_4bit=True requires `bitsandbytes`, which is not installed. "
                    "Install it with `pip install bitsandbytes`, or construct this client with "
                    "load_in_4bit=False (uses more memory, works on CPU-only machines)."
                )
            from transformers import BitsAndBytesConfig

            quantisation_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=getattr(torch, dtype),
                **quantisation_kwargs,
            )
        except OSError as exc:
            raise DependencyError(
                f"Could not load base model '{model_id}': {exc}. If this is a Hugging Face Hub id, "
                f"confirm network access and that the weights are downloaded/cached; if it is a "
                f"local path, confirm it exists."
            ) from exc

        for composition in adapters:
            from peft import PeftModel

            try:
                model = PeftModel.from_pretrained(model, composition.local_path, is_trainable=False)
            except OSError as exc:
                raise DependencyError(
                    f"Could not load LoRA adapter '{composition.ref.uri}' from "
                    f"{composition.local_path}: {exc}. This path is expected to come from P4's "
                    f"Registry `load` endpoint (Week 7); it must exist on disk before evaluation."
                ) from exc

        model.eval()
        _LOG.info(
            "TransformersEdgeInferenceClient loaded %s (%d adapter(s) composed)", model_id, len(adapters)
        )
        return model, tokenizer


def _is_importable(module_name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def missing_dependencies(
    *, adapters: Sequence[LoraComposition] = (), load_in_4bit: bool = True
) -> list[str]:
    """Which packages a ``TransformersEdgeInferenceClient(...)`` call shaped
    like ``(adapters=adapters, load_in_4bit=load_in_4bit)`` would need but lack.

    Pure preflight: only calls :func:`importlib.util.find_spec`, so it is
    safe to call from a test, a dependency-preflight script, or a CLI
    without importing torch/transformers/peft/bitsandbytes or any side
    effects. Mirrors exactly the requirement set :meth:`TransformersEdgeInferenceClient._load`
    enforces at construction time — ``torch``/``transformers``
    unconditionally, ``peft`` only when ``adapters`` is non-empty,
    ``bitsandbytes`` only when ``load_in_4bit`` is true — so callers do not
    need to (and must not) hand-duplicate that logic, which would drift the
    moment the real requirement set changes.

    A single-package check (e.g. "is torch installed?") is not sufficient to
    answer "would this construct successfully?": on a machine with ``torch``
    installed but not ``transformers`` (or with both installed but not
    ``bitsandbytes`` when ``load_in_4bit=True``), construction still raises
    ``DependencyError`` — this function is what lets a caller tell the two
    situations apart without guessing.
    """
    missing = [pkg for pkg in ("torch", "transformers") if not _is_importable(pkg)]
    if adapters and not _is_importable("peft"):
        missing.append("peft")
    if load_in_4bit and not _is_importable("bitsandbytes"):
        missing.append("bitsandbytes")
    return missing


def register_transformers_edge_client(
    model_id: str,
    *,
    adapters: Sequence[LoraComposition] = (),
    load_in_4bit: bool = True,
) -> None:
    """Register :class:`TransformersEdgeInferenceClient` as the ``edge`` backend.

    Calls :func:`evaluation.registry.register_edge_client_factory` with a
    closure over the given model/adapter configuration, so that flipping
    ``configs/evaluation.yaml``'s ``backend.kind`` to ``"edge"`` is
    sufficient to switch the whole harness — HumanEval and MBPP both go
    through :class:`~evaluation.registry.build_inference_client`, so wiring
    happens exactly once here rather than per-benchmark.

    This is the concrete answer to Week-3 Monday/Tuesday ("wire the
    HumanEval/MBPP runner to call the merged model") once real weights and
    adapters are available; see the module docstring for why it is not
    exercised end-to-end in this repository today.
    """
    from evaluation.registry import register_edge_client_factory

    def _factory(_backend_config) -> TransformersEdgeInferenceClient:
        return TransformersEdgeInferenceClient(model_id, adapters=adapters, load_in_4bit=load_in_4bit)

    register_edge_client_factory(_factory)
