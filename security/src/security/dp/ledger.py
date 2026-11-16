"""The privacy budget of one client over one federated run (D7, U3).

A run is ``RUN_ROUNDS`` rounds and may spend at most ``RUN_EPSILON_BUDGET`` at
``RUN_DELTA``. Every uploaded round is one :class:`RoundRecord`; the cumulative
epsilon is read from ONE accountant that replays every record's
``(noise_multiplier, sample_rate, steps)``. Per-round epsilons are never added
up: under the PRV accountant two rounds compose to much less than their sum.

The ledger is charged when an adapter is uploaded, not when it is trained (C8):
the privacy cost is incurred by what leaves the laptop.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .accountant import DEFAULT_ACCOUNTANT, BudgetExhaustedError

try:
    from opacus.accountants import create_accountant
except ImportError:  # pragma: no cover - opacus is a hard dependency of the package
    create_accountant = None

RUN_EPSILON_BUDGET = 8.0
RUN_DELTA = 1e-5
RUN_ROUNDS = 2

#: Slack on the budget comparison: accountant output is floating point.
BUDGET_TOLERANCE = 1e-6

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class LedgerError(Exception):
    """The ledger file exists but cannot be trusted (unreadable or malformed)."""


@dataclass(frozen=True)
class RoundRecord:
    round: int                # 1-based position in the run
    noise_multiplier: float
    sample_rate: float
    steps: int
    epsilon_round: float      # this round alone, as the training manifest reported it
    adapter_sha256: str       # what was uploaded
    uploaded_at: str


def _compose(history: list[tuple[float, float, int]], delta: float, accountant: str) -> float:
    """Epsilon at ``delta`` after every ``(sigma, sample_rate, steps)`` in order."""
    if create_accountant is None:  # pragma: no cover
        raise ImportError("Opacus is required for the privacy ledger.")
    if not history:
        return 0.0
    acc = create_accountant(accountant)
    for sigma, rate, steps in history:
        for _ in range(int(steps)):
            acc.step(noise_multiplier=float(sigma), sample_rate=float(rate))
    return float(acc.get_epsilon(delta=delta))


class BudgetLedger:
    """One client's spend in one run, kept in a JSON file."""

    def __init__(self, path: Path, *, budget: float = RUN_EPSILON_BUDGET, delta: float = RUN_DELTA,
                 rounds_total: int = RUN_ROUNDS, accountant: str = DEFAULT_ACCOUNTANT) -> None:
        self.path = Path(path)
        self.budget = float(budget)
        self.delta = float(delta)
        self.rounds_total = int(rounds_total)
        self.accountant = accountant
        self._records: list[RoundRecord] = self._load()

    def _load(self) -> list[RoundRecord]:
        if not self.path.exists():
            return []
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
            return [RoundRecord(**entry) for entry in doc["records"]]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise LedgerError(
                f"privacy ledger {self.path} is unreadable ({exc}); it was left untouched. "
                "Restore it from a backup, or start a new run with a new run id."
            ) from exc

    def records(self) -> list[RoundRecord]:
        return list(self._records)

    def has(self, adapter_sha256: str) -> bool:
        return any(r.adapter_sha256 == adapter_sha256 for r in self._records)

    def cumulative_epsilon(self, extra: RoundRecord | None = None) -> float:
        records = self._records + ([extra] if extra is not None else [])
        return _compose(
            [(r.noise_multiplier, r.sample_rate, r.steps) for r in records],
            self.delta, self.accountant,
        )

    def check(self, extra: RoundRecord) -> None:
        """Raise :class:`BudgetExhaustedError` when ``extra`` does not fit in the run."""
        used = len(self._records)
        if used >= self.rounds_total:
            raise BudgetExhaustedError(
                f"run already has {used} of {self.rounds_total} rounds "
                f"(epsilon {self.cumulative_epsilon():.2f} of {self.budget:g} spent)"
            )
        total = self.cumulative_epsilon(extra)
        if total > self.budget + BUDGET_TOLERANCE:
            raise BudgetExhaustedError(
                f"this round would bring the run to epsilon {total:.2f}, over the budget of "
                f"{self.budget:g} (epsilon {self.cumulative_epsilon():.2f} spent in {used} "
                f"of {self.rounds_total} rounds)"
            )

    def charge(self, record: RoundRecord) -> float:
        """Check, append and fsync. Returns the new cumulative epsilon.

        Charging an adapter that is already in the ledger changes nothing.
        """
        if self.has(record.adapter_sha256):
            return self.cumulative_epsilon()
        self.check(record)
        records = [*self._records, record]
        self._write(records)
        self._records = records
        return self.cumulative_epsilon()

    def _write(self, records: list[RoundRecord]) -> None:
        doc = {
            "client_id": self.path.parent.name,
            "run_id": self.path.stem,
            "budget": self.budget,
            "delta": self.delta,
            "rounds_total": self.rounds_total,
            "records": [asdict(r) for r in records],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)


def ledger_path(root: Path, client_id: str, run_id: str) -> Path:
    """``<root>/<client_id>/<run_id>.json``; both names must be plain file names."""
    for label, value in (("client id", client_id), ("run id", run_id)):
        if not _NAME.match(value or ""):
            raise ValueError(f"{label} {value!r} must be letters, digits, '.', '_' or '-'")
    return Path(root) / client_id / f"{run_id}.json"


def calibrate_run_sigma(*, steps_per_round: int, sample_rate: float, rounds_total: int = RUN_ROUNDS,
                        budget: float = RUN_EPSILON_BUDGET, delta: float = RUN_DELTA,
                        accountant: str = DEFAULT_ACCOUNTANT) -> float:
    """Smallest noise multiplier for which ``rounds_total`` identical rounds fit ``budget`` (C9)."""
    if steps_per_round < 1 or rounds_total < 1:
        raise ValueError("steps_per_round and rounds_total must be at least 1")
    if not 0 < sample_rate <= 1:
        raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")

    def fits(sigma: float) -> bool:
        history = [(sigma, sample_rate, steps_per_round * rounds_total)]
        return _compose(history, delta, accountant) <= budget

    low, high = 0.0, 1.0
    while not fits(high):
        low, high = high, high * 2
        if high > 4096:
            raise ValueError(f"no noise multiplier up to 4096 fits epsilon {budget:g}")
    while high - low > 0.005:
        mid = (low + high) / 2
        if fits(mid):
            high = mid
        else:
            low = mid
    return high
