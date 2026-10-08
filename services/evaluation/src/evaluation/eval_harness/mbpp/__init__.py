"""MBPP benchmark support.

Contents:

* ``adapter.py`` — :class:`~eval_harness.mbpp.adapter.MbppAdapter`.
* ``sample_tasks.jsonl`` — five original, format-compatible tasks used for
  the offline Week-1 dry run. Not the published MBPP dataset.
"""

from evaluation.eval_harness.mbpp.adapter import MbppAdapter

__all__ = ["MbppAdapter"]
