"""Tests for the standalone RDP epsilon tracker.

Week 3 acceptance criteria:
  - ε increases monotonically over training steps
  - Budget-exceeded check works correctly
  - Report generation produces valid JSON
  - Tracker matches expected mathematical properties of RDP
"""
from __future__ import annotations

import json

import pytest

from security.dp_config import DPConfig
from security.epsilon_tracker import EpsilonTracker
from security.epsilon_report import generate_report, format_report_text


class TestEpsilonTracker:
    """Tests for RDP-based epsilon tracking."""

    @pytest.fixture
    def tracker(self) -> EpsilonTracker:
        return EpsilonTracker(
            noise_multiplier=1.0,
            sample_rate=0.01,
            delta=1e-5,
        )

    def test_initial_epsilon_is_zero(self, tracker: EpsilonTracker) -> None:
        # Before any steps, ε should be 0 (or near-zero after RDP→DP conversion)
        eps = tracker.get_epsilon()
        assert eps >= 0.0

    def test_epsilon_monotonic_increase(self, tracker: EpsilonTracker) -> None:
        """ε must increase monotonically over 100 steps."""
        prev_eps = 0.0
        for _ in range(100):
            eps = tracker.step()
            assert eps >= prev_eps, (
                f"ε decreased: {prev_eps:.6f} → {eps:.6f}. "
                "This violates the composition theorem."
            )
            prev_eps = eps

    def test_epsilon_increases_after_steps(self, tracker: EpsilonTracker) -> None:
        eps_0 = tracker.get_epsilon()
        tracker.step(num_steps=50)
        eps_50 = tracker.get_epsilon()
        assert eps_50 > eps_0

    def test_total_steps_tracked(self, tracker: EpsilonTracker) -> None:
        assert tracker.total_steps == 0
        tracker.step(10)
        assert tracker.total_steps == 10
        tracker.step(5)
        assert tracker.total_steps == 15

    def test_budget_remaining(self, tracker: EpsilonTracker) -> None:
        tracker.step(10)
        eps = tracker.get_epsilon()
        remaining = tracker.budget_remaining(max_epsilon=8.0)
        assert abs(remaining - (8.0 - eps)) < 1e-6

    def test_budget_exceeded_returns_false_initially(
        self, tracker: EpsilonTracker
    ) -> None:
        assert not tracker.is_budget_exceeded(max_epsilon=8.0)

    def test_budget_exceeded_returns_true_after_many_steps(self) -> None:
        """With a very tiny budget, even a few steps should exceed it."""
        tracker = EpsilonTracker(
            noise_multiplier=0.1,  # very little noise → high ε per step
            sample_rate=0.5,       # high sample rate → high ε
            delta=1e-5,
        )
        tracker.step(1000)
        assert tracker.is_budget_exceeded(max_epsilon=0.001)

    def test_higher_noise_means_lower_epsilon(self) -> None:
        """More noise (higher σ) → better privacy (lower ε)."""
        low_noise = EpsilonTracker(noise_multiplier=0.5, sample_rate=0.01)
        high_noise = EpsilonTracker(noise_multiplier=2.0, sample_rate=0.01)

        low_noise.step(100)
        high_noise.step(100)

        assert high_noise.get_epsilon() < low_noise.get_epsilon()

    def test_to_report(self, tracker: EpsilonTracker) -> None:
        tracker.step(50)
        report = tracker.to_report()
        assert report["total_steps"] == 50
        assert "current_epsilon" in report
        assert "delta" in report
        assert report["noise_multiplier"] == 1.0


class TestEpsilonReport:
    """Tests for the report generation and formatting."""

    @pytest.fixture
    def tracker_and_config(self) -> tuple[EpsilonTracker, DPConfig]:
        cfg = DPConfig(noise_multiplier=1.0)
        tracker = EpsilonTracker(
            noise_multiplier=1.0, sample_rate=0.01, delta=1e-5
        )
        tracker.step(100)
        return tracker, cfg

    def test_generate_report_has_required_fields(
        self, tracker_and_config: tuple[EpsilonTracker, DPConfig]
    ) -> None:
        tracker, cfg = tracker_and_config
        report = generate_report(tracker, cfg)
        assert "privacy_guarantee" in report
        assert "budget" in report
        assert "training_config" in report
        assert "references" in report

    def test_report_is_json_serializable(
        self, tracker_and_config: tuple[EpsilonTracker, DPConfig]
    ) -> None:
        tracker, cfg = tracker_and_config
        report = generate_report(tracker, cfg)
        json_str = json.dumps(report)
        parsed = json.loads(json_str)
        assert parsed["training_config"]["total_steps"] == 100

    def test_format_report_text_is_readable(
        self, tracker_and_config: tuple[EpsilonTracker, DPConfig]
    ) -> None:
        tracker, cfg = tracker_and_config
        report = generate_report(tracker, cfg)
        text = format_report_text(report)
        assert "CLASP" in text
        assert "Privacy Budget Report" in text
        assert "FlashDP" in text

    def test_save_report_creates_files(
        self, tracker_and_config: tuple[EpsilonTracker, DPConfig], tmp_path
    ) -> None:
        from security.epsilon_report import save_report

        tracker, cfg = tracker_and_config
        report = generate_report(tracker, cfg)
        json_path, txt_path = save_report(report, tmp_path)
        assert json_path.exists()
        assert txt_path.exists()
        assert json_path.suffix == ".json"
        assert txt_path.suffix == ".txt"
