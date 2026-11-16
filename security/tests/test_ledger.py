"""Tests for security.dp.ledger: the per-client privacy budget over a run."""
from __future__ import annotations

import json

import pytest

opacus = pytest.importorskip("opacus", reason="opacus required for ledger tests")

from opacus.accountants import create_accountant  # noqa: E402 — after the importorskip guard
from opacus.accountants.utils import get_noise_multiplier  # noqa: E402

from security.dp.accountant import DEFAULT_ACCOUNTANT, BudgetExhaustedError  # noqa: E402
from security.dp.ledger import (  # noqa: E402
    RUN_DELTA,
    RUN_EPSILON_BUDGET,
    RUN_ROUNDS,
    BudgetLedger,
    LedgerError,
    RoundRecord,
    calibrate_run_sigma,
    ledger_path,
)

STEPS = 160
RATE = 1 / 160


def _round_epsilon(sigma: float, rate: float, steps: int) -> float:
    acc = create_accountant(DEFAULT_ACCOUNTANT)
    for _ in range(steps):
        acc.step(noise_multiplier=sigma, sample_rate=rate)
    return float(acc.get_epsilon(delta=RUN_DELTA))


def _rec(round_no: int, sigma: float, rate: float, steps: int, sha: str | None = None) -> RoundRecord:
    return RoundRecord(
        round=round_no,
        noise_multiplier=sigma,
        sample_rate=rate,
        steps=steps,
        epsilon_round=_round_epsilon(sigma, rate, steps),
        adapter_sha256=sha or f"{round_no:064x}",
        uploaded_at=f"2026-11-16T00:00:0{round_no}+00:00",
    )


@pytest.fixture(scope="module")
def run_sigma() -> float:
    return calibrate_run_sigma(steps_per_round=STEPS, sample_rate=RATE)


def test_the_spec_numbers() -> None:
    assert (RUN_EPSILON_BUDGET, RUN_DELTA, RUN_ROUNDS) == (8.0, 1e-5, 2)


def test_two_calibrated_rounds_fit_and_one_alone_is_below_budget(tmp_path, run_sigma) -> None:
    led = BudgetLedger(tmp_path / "l.json")
    e1 = led.charge(_rec(1, run_sigma, RATE, STEPS))
    e2 = led.charge(_rec(2, run_sigma, RATE, STEPS))
    assert e1 < e2 <= 8.0 + 1e-6 and e2 > 7.5  # tight, not wasteful


def test_composition_is_not_a_sum(tmp_path, run_sigma) -> None:
    led = BudgetLedger(tmp_path / "l.json")
    r1, r2 = _rec(1, run_sigma, RATE, STEPS), _rec(2, run_sigma, RATE, STEPS)
    led.charge(r1)
    cumulative = led.charge(r2)
    assert cumulative < r1.epsilon_round + r2.epsilon_round - 0.5
    assert cumulative > max(r1.epsilon_round, r2.epsilon_round)


def test_cumulative_with_an_extra_record_does_not_write(tmp_path, run_sigma) -> None:
    led = BudgetLedger(tmp_path / "l.json")
    r1 = _rec(1, run_sigma, RATE, STEPS)
    assert led.cumulative_epsilon() == 0.0
    assert led.cumulative_epsilon(r1) == pytest.approx(r1.epsilon_round, abs=1e-6)
    assert led.records() == []
    assert not (tmp_path / "l.json").exists()


def test_third_round_is_refused_and_nothing_is_written(tmp_path, run_sigma) -> None:
    path = tmp_path / "l.json"
    led = BudgetLedger(path)
    led.charge(_rec(1, run_sigma, RATE, STEPS))
    led.charge(_rec(2, run_sigma, RATE, STEPS))
    before = path.read_bytes()
    # even a round with enormous noise (almost free) is refused: the run is over
    with pytest.raises(BudgetExhaustedError, match="2 of 2 rounds"):
        led.charge(_rec(3, 50.0, RATE, STEPS))
    assert path.read_bytes() == before
    assert len(led.records()) == 2


def test_a_round_trained_for_epsilon_8_alone_makes_round_two_impossible(tmp_path) -> None:
    # What train_dp did before: each round calibrated to the whole budget.
    alone = get_noise_multiplier(
        target_epsilon=8.0, target_delta=RUN_DELTA, sample_rate=RATE, steps=STEPS,
        accountant=DEFAULT_ACCOUNTANT,
    )
    path = tmp_path / "l.json"
    led = BudgetLedger(path)
    led.charge(_rec(1, alone, RATE, STEPS))
    before = path.read_bytes()
    with pytest.raises(BudgetExhaustedError, match="8"):
        led.charge(_rec(2, alone, RATE, STEPS))
    assert path.read_bytes() == before
    assert len(led.records()) == 1


def test_ledger_survives_reopen_and_has_finds_the_sha(tmp_path, run_sigma) -> None:
    path = tmp_path / "deep" / "l.json"
    led = BudgetLedger(path)
    r1 = _rec(1, run_sigma, RATE, STEPS, sha="ab" * 32)
    e1 = led.charge(r1)

    again = BudgetLedger(path)
    assert again.records() == [r1]
    assert again.has("ab" * 32) and not again.has("cd" * 32)
    assert again.cumulative_epsilon() == pytest.approx(e1, abs=1e-9)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["budget"] == 8.0 and doc["delta"] == 1e-5 and doc["rounds_total"] == 2
    assert len(doc["records"]) == 1
    assert not list(path.parent.glob("*.tmp"))


def test_charging_the_same_adapter_twice_charges_once(tmp_path, run_sigma) -> None:
    led = BudgetLedger(tmp_path / "l.json")
    r1 = _rec(1, run_sigma, RATE, STEPS, sha="ab" * 32)
    e1 = led.charge(r1)
    assert led.charge(r1) == pytest.approx(e1, abs=1e-9)
    assert len(led.records()) == 1


def test_corrupt_ledger_file_raises_and_is_not_overwritten(tmp_path, run_sigma) -> None:
    path = tmp_path / "l.json"
    path.write_text('{"records": [{"round": 1', encoding="utf-8")
    with pytest.raises(LedgerError, match="l.json"):
        BudgetLedger(path)
    assert path.read_text(encoding="utf-8") == '{"records": [{"round": 1'


def test_ledger_path_is_per_client_and_run(tmp_path) -> None:
    assert ledger_path(tmp_path, "client-flask", "run-1") == tmp_path / "client-flask" / "run-1.json"
    for bad in ("../x", "a/b", "", "a\\b"):
        with pytest.raises(ValueError):
            ledger_path(tmp_path, "client-flask", bad)
        with pytest.raises(ValueError):
            ledger_path(tmp_path, bad, "run-1")


def test_calibrated_sigma_is_the_smallest_that_fits(run_sigma) -> None:
    def two_rounds(sigma: float) -> float:
        return _round_epsilon(sigma, RATE, 2 * STEPS)

    assert two_rounds(run_sigma) <= 8.0 + 1e-6
    assert two_rounds(run_sigma - 0.02) > 8.0
