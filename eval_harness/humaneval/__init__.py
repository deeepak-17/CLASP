"""HumanEval benchmark support.

Contents:

* ``adapter.py`` — :class:`~eval_harness.humaneval.adapter.HumanEvalAdapter`.
* ``sample_tasks.jsonl`` — five original, format-compatible tasks used for
  the offline Week-1 dry run. Not the published HumanEval dataset.
"""

from eval_harness.humaneval.adapter import HumanEvalAdapter

__all__ = ["HumanEvalAdapter"]
