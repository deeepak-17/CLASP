"""HumanEval adapter.

HumanEval (Chen et al., 2021) gives the model a function signature plus
docstring and asks it to produce the body. The prompt is therefore used
verbatim, and a completion is appended directly beneath it.

Task source
-----------
``configs/evaluation.yaml`` points ``tasks_path`` at a JSONL file in the
canonical HumanEval field layout. When that file is absent the adapter falls
back to ``sample_tasks.jsonl``, a small set of **original** tasks written for
this repository in a format-compatible layout. They exist so the Week-1 dry
run works with no download and no network; they are not the published
benchmark and numbers from them are not comparable to published results.

Fetching the real dataset is a Week-3 concern
(``docs/eval_harness.md`` records the two supported routes).
"""

from __future__ import annotations

from evaluation.base import BenchmarkAdapter
from evaluation.models import EvalTask
from interfaces.contracts import BenchmarkName


class HumanEvalAdapter(BenchmarkAdapter):
    """Adapter for HumanEval-style function-completion tasks."""

    name = BenchmarkName.HUMANEVAL

    def build_prompt(self, task: EvalTask) -> str:
        """Return the HumanEval prompt unchanged.

        HumanEval prompts are already complete Python source ending inside a
        function body, so any added framing would change the distribution the
        published numbers were measured under.
        """
        return task.prompt

    def assemble_program(self, task: EvalTask, completion: str) -> str:
        """Build ``prompt + completion + tests + check(entry_point)``.

        This is the exact program shape HumanEval's own evaluation harness
        executes, reproduced here so Week-3 scoring only has to add the
        sandbox, not redesign the assembly.
        """
        prompt = task.prompt if task.prompt.endswith("\n") else task.prompt + "\n"
        body = completion if completion.endswith("\n") else completion + "\n"
        return f"{prompt}{body}\n{task.test_code}\ncheck({task.entry_point})\n"
