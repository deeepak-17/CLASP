"""HumanEval benchmark support.

Contents:

* ``adapter.py`` — :class:`~evaluation.humaneval.adapter.HumanEvalAdapter`.
* ``sample_tasks.jsonl`` — five original, format-compatible tasks used for
  the offline Week-1 dry run. Not the published HumanEval dataset.
"""

from evaluation.humaneval.adapter import HumanEvalAdapter

__all__ = ["HumanEvalAdapter"]
