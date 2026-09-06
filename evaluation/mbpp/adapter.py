"""MBPP adapter.

MBPP (Austin et al., 2021) states each problem in natural language and
supplies a list of asserts. Unlike HumanEval there is no function signature in
the prompt, so the adapter must synthesise one — and it must do so in a way
that pins the function *name*, because the asserts call it by name. Handing
the model prose alone would make failures indistinguishable between "wrong
logic" and "named the function something else".

The convention used here is the standard MBPP 3-shot-free framing: the
description, the first assert as a worked specification example, and a stub
``def <entry_point>(...)`` line the model completes.
"""

from __future__ import annotations

from evaluation.base import BenchmarkAdapter
from evaluation.models import EvalTask
from interfaces.contracts import BenchmarkName


class MbppAdapter(BenchmarkAdapter):
    """Adapter for MBPP-style natural-language programming tasks."""

    name = BenchmarkName.MBPP

    def build_prompt(self, task: EvalTask) -> str:
        """Build a docstring-framed prompt that fixes the function name.

        Including the first assert (controlled by
        ``benchmarks.mbpp.include_test_example_in_prompt``) is standard MBPP
        practice: it disambiguates argument order and return type, which the
        prose alone frequently leaves open.
        """
        lines = ['"""', task.prompt.strip()]

        if self._config.include_test_example_in_prompt:
            example = self._first_assert(task)
            if example:
                lines += ["", f"Example: {example}"]

        lines += ['"""', "", ""]
        return "\n".join(lines) + self._signature(task)

    def assemble_program(self, task: EvalTask, completion: str) -> str:
        """Build ``prompt + completion + asserts``.

        MBPP has no ``check()`` wrapper; the asserts run at module level.
        """
        prompt = self.build_prompt(task)
        body = completion if completion.endswith("\n") else completion + "\n"
        return f"{prompt}{body}\n{task.test_code}"

    # --- internals ---------------------------------------------------------
    @staticmethod
    def _signature(task: EvalTask) -> str:
        """Return the ``def`` line the model is asked to complete.

        A signature recorded in metadata is preferred; otherwise a permissive
        ``*args`` stub is emitted, which still pins the name.

        The signature line itself is NOT the completion; it's part of the
        prompt that establishes the function name. The model completes from
        the line after the colon.
        """
        signature = task.metadata.get("signature")
        if isinstance(signature, str) and signature.strip():
            line = signature.strip()
            return (line if line.endswith(":") else f"{line}:") + "\n"
        return f"def {task.entry_point}(*args):\n"

    @staticmethod
    def _first_assert(task: EvalTask) -> str | None:
        """First assert statement from ``test_list`` metadata or ``test_code``."""
        test_list = task.metadata.get("test_list")
        if isinstance(test_list, list) and test_list:
            return str(test_list[0]).strip()
        for line in task.test_code.splitlines():
            stripped = line.strip()
            if stripped.startswith("assert "):
                return stripped
        return None
