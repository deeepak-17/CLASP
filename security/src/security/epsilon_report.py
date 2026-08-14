"""Privacy-budget report formatter for panel sign-off.

Generates both JSON and human-readable reports summarising the current
differential privacy guarantee after training.

The report is designed for inclusion in the threat model appendix and
for panel review (23CSE498 project evaluation).
"""
from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Any

from security.dp_config import DPConfig
from security.epsilon_tracker import EpsilonTracker


def generate_report(
    tracker: EpsilonTracker,
    dp_config: DPConfig,
    max_epsilon: float = 8.0,
    dataset_size: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a comprehensive privacy-budget report.

    Returns a dict suitable for JSON serialisation or pretty-printing.
    """
    eps = tracker.get_epsilon()
    remaining = tracker.budget_remaining(max_epsilon)
    exceeded = tracker.is_budget_exceeded(max_epsilon)

    report: dict[str, Any] = {
        "report_type": "CLASP Privacy Budget Report",
        "generated_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "privacy_guarantee": {
            "current_epsilon": round(eps, 6),
            "delta": tracker.delta,
            "guarantee": f"({eps:.4f}, {tracker.delta})-DP",
        },
        "budget": {
            "max_epsilon": max_epsilon,
            "remaining": round(remaining, 6),
            "exceeded": exceeded,
            "status": "🔴 EXCEEDED" if exceeded else "🟢 WITHIN BUDGET",
        },
        "training_config": {
            "total_steps": tracker.total_steps,
            "noise_multiplier": tracker.noise_multiplier,
            "sample_rate": round(tracker.sample_rate, 6),
            "max_grad_norm": dp_config.max_grad_norm,
            "grad_sample_mode": dp_config.grad_sample_mode,
            "accountant_type": dp_config.accountant_type,
        },
        "references": [
            "FlashDP — Wang et al., NeurIPS 2025",
            "DP-SGD-RC — Ullah et al., ICML 2026",
            "Tighter Privacy Auditing — Cebere et al., ICLR 2025",
            "Time-Adaptive Privacy — Kiani et al., ICLR 2025",
        ],
    }

    if dataset_size is not None:
        report["training_config"]["dataset_size"] = dataset_size

    if extra_metadata:
        report["metadata"] = extra_metadata

    return report


def format_report_text(report: dict[str, Any]) -> str:
    """Render a human-readable text version of the privacy report."""
    lines = [
        "=" * 60,
        "  CLASP — Privacy Budget Report",
        "=" * 60,
        "",
        f"  Generated: {report['generated_utc']}",
        "",
        "  PRIVACY GUARANTEE",
        f"    Current ε    : {report['privacy_guarantee']['current_epsilon']}",
        f"    δ            : {report['privacy_guarantee']['delta']}",
        f"    Guarantee    : {report['privacy_guarantee']['guarantee']}",
        "",
        "  BUDGET STATUS",
        f"    Max ε        : {report['budget']['max_epsilon']}",
        f"    Remaining ε  : {report['budget']['remaining']}",
        f"    Status       : {report['budget']['status']}",
        "",
        "  TRAINING CONFIG",
        f"    Total Steps  : {report['training_config']['total_steps']}",
        f"    σ (noise)    : {report['training_config']['noise_multiplier']}",
        f"    Sample Rate  : {report['training_config']['sample_rate']}",
        f"    Clip Norm    : {report['training_config']['max_grad_norm']}",
        f"    Clipping     : {report['training_config']['grad_sample_mode']}",
        f"    Accountant   : {report['training_config']['accountant_type']}",
        "",
        "  REFERENCES",
    ]
    for ref in report.get("references", []):
        lines.append(f"    • {ref}")
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def save_report(
    report: dict[str, Any],
    output_dir: str | Path,
    basename: str = "epsilon_budget_report",
) -> tuple[Path, Path]:
    """Save report as both JSON and plain-text files.

    Returns (json_path, txt_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{basename}.json"
    txt_path = output_dir / f"{basename}.txt"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    txt_path.write_text(format_report_text(report), encoding="utf-8")

    return json_path, txt_path
