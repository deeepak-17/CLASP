"""Dependency-free SVG charts for report figures.

Report figures are regenerated from committed result files, so they must be
deterministic and must not need a plotting stack in CI. Two forms cover the
report: grouped bars (magnitude or signed values, one series per group
member) and lines (a quantity over a numeric x). Both draw from a fixed
categorical palette in order, put the legend above the plot, keep gridlines
and axes recessive, and render text in neutral ink rather than series colors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence
from xml.sax.saxutils import escape

#: Categorical slots in fixed order (the dashboard's validated pair, then two more).
PALETTE = ("#2a78d6", "#eb6834", "#1c9e77", "#8a5cc2")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


@dataclass(frozen=True)
class Series:
    name: str
    values: Sequence[float | None]


def _nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if hi == lo:
        hi = lo + 1.0
    raw = (hi - lo) / count
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 1e-9:
        ticks.append(round(v, 10) + 0.0)  # + 0.0 turns -0.0 into 0.0, so the label is "0" not "-0"
        v += step
    if ticks[-1] < hi:
        ticks.append(round(ticks[-1] + step, 10))
    return ticks


def _text(x: float, y: float, s: str, *, size: int = 12, fill: str = INK_SECONDARY, anchor: str = "middle",
          weight: int = 400) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{escape(s)}</text>')


def _frame(width: int, height: int, title: str, subtitle: str | None, body: list[str]) -> str:
    head = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{FONT}" role="img" aria-label="{escape(title)}">',
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
        _text(16, 26, title, size=15, fill=INK, anchor="start", weight=600),
    ]
    if subtitle:
        head.append(_text(16, 46, subtitle, size=12, fill=INK_SECONDARY, anchor="start"))
    return "\n".join([*head, *body, "</svg>"]) + "\n"


def _legend(series: Sequence[Series], x: float, y: float) -> list[str]:
    out, cx = [], x
    for i, s in enumerate(series):
        out.append(f'<rect x="{cx:.1f}" y="{y - 9:.1f}" width="10" height="10" rx="2" fill="{PALETTE[i]}"/>')
        out.append(_text(cx + 15, y, s.name, size=12, fill=INK_SECONDARY, anchor="start"))
        cx += 30 + 7 * len(s.name)
    return out


def grouped_bars(
    categories: Sequence[str],
    series: Sequence[Series],
    *,
    title: str,
    subtitle: str | None = None,
    y_label: str = "",
    fmt: Callable[[float], str] = lambda v: f"{v:g}",
    width: int = 760,
    height: int = 380,
) -> str:
    """Grouped bar chart; bars start at zero, signed values extend either side of it."""
    if len(series) > len(PALETTE):
        raise ValueError(f"at most {len(PALETTE)} series")
    values = [v for s in series for v in s.values if v is not None]
    lo, hi = min(0.0, *values), max(0.0, *values)
    ticks = _nice_ticks(lo, hi)
    y_lo, y_hi = ticks[0], ticks[-1]
    left, right, top, bottom = 64, 16, 80, 44
    pw, ph = width - left - right, height - top - bottom

    def y(v: float) -> float:
        return top + ph * (y_hi - v) / (y_hi - y_lo)

    body = _legend(series, left, 66)
    for t in ticks:
        body.append(f'<line x1="{left}" x2="{left + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="{GRID}"/>')
        body.append(_text(left - 8, y(t) + 4, fmt(t), size=11, fill=INK_MUTED, anchor="end"))
    if y_label:
        body.append(f'<text transform="translate(14 {top + ph / 2:.1f}) rotate(-90)" font-size="11" '
                    f'fill="{INK_MUTED}" text-anchor="middle">{escape(y_label)}</text>')
    group = pw / max(1, len(categories))
    bar = min(28.0, group * 0.76 / len(series))
    for ci, cat in enumerate(categories):
        gx = left + group * ci + (group - bar * len(series) - 2 * (len(series) - 1)) / 2
        for si, s in enumerate(series):
            v = s.values[ci]
            if v is None:
                continue
            x0 = gx + si * (bar + 2)
            y0, y1 = sorted((y(0.0), y(v)))
            h = max(y1 - y0, 0.5)
            r = min(4.0, h / 2, bar / 2)
            # Rounded only at the data end, flat on the zero line.
            if v >= 0:
                path = (f"M{x0:.1f},{y1:.1f} V{y0 + r:.1f} Q{x0:.1f},{y0:.1f} {x0 + r:.1f},{y0:.1f} "
                        f"H{x0 + bar - r:.1f} Q{x0 + bar:.1f},{y0:.1f} {x0 + bar:.1f},{y0 + r:.1f} V{y1:.1f} Z")
            else:
                path = (f"M{x0:.1f},{y0:.1f} V{y1 - r:.1f} Q{x0:.1f},{y1:.1f} {x0 + r:.1f},{y1:.1f} "
                        f"H{x0 + bar - r:.1f} Q{x0 + bar:.1f},{y1:.1f} {x0 + bar:.1f},{y1 - r:.1f} V{y0:.1f} Z")
            body.append(f'<path d="{path}" fill="{PALETTE[si]}"><title>{escape(f"{cat} · {s.name}: {fmt(v)}")}'
                        "</title></path>")
        body.append(_text(left + group * ci + group / 2, top + ph + 18, cat, size=12, fill=INK_SECONDARY))
    body.append(f'<line x1="{left}" x2="{left + pw}" y1="{y(0.0):.1f}" y2="{y(0.0):.1f}" stroke="{AXIS}"/>')
    return _frame(width, height, title, subtitle, body)


def lines(
    xs: Sequence[float],
    series: Sequence[Series],
    *,
    title: str,
    subtitle: str | None = None,
    x_label: str = "",
    y_label: str = "",
    fmt: Callable[[float], str] = lambda v: f"{v:g}",
    reference: tuple[float, str] | None = None,
    log_x: bool = False,
    width: int = 760,
    height: int = 380,
) -> str:
    """Line chart over numeric x, with an optional labelled horizontal reference line."""
    if len(series) > len(PALETTE):
        raise ValueError(f"at most {len(PALETTE)} series")
    values = [v for s in series for v in s.values if v is not None]
    if reference:
        values.append(reference[0])
    ticks = _nice_ticks(0.0, max(values))
    y_hi = ticks[-1]
    left, right, top, bottom = 64, 16, 80, 52
    pw, ph = width - left - right, height - top - bottom
    tx = (lambda v: math.log10(v)) if log_x else (lambda v: v)
    x_lo, x_hi = tx(min(xs)), tx(max(xs))

    def px(v: float) -> float:
        return left + pw * (tx(v) - x_lo) / (x_hi - x_lo)

    def py(v: float) -> float:
        return top + ph * (y_hi - v) / y_hi

    body = _legend(series, left, 66)
    for t in ticks:
        body.append(f'<line x1="{left}" x2="{left + pw}" y1="{py(t):.1f}" y2="{py(t):.1f}" stroke="{GRID}"/>')
        body.append(_text(left - 8, py(t) + 4, fmt(t), size=11, fill=INK_MUTED, anchor="end"))
    for xv in xs:
        body.append(_text(px(xv), top + ph + 18, f"{xv:g}", size=11, fill=INK_MUTED))
    if x_label:
        body.append(_text(left + pw / 2, height - 10, x_label, size=11, fill=INK_MUTED))
    if y_label:
        body.append(f'<text transform="translate(14 {top + ph / 2:.1f}) rotate(-90)" font-size="11" '
                    f'fill="{INK_MUTED}" text-anchor="middle">{escape(y_label)}</text>')
    if reference:
        ry = py(reference[0])
        body.append(f'<line x1="{left}" x2="{left + pw}" y1="{ry:.1f}" y2="{ry:.1f}" stroke="{INK_SECONDARY}" '
                    'stroke-dasharray="4 3"/>')
        body.append(_text(left + 6, ry - 6, reference[1], size=11, fill=INK_SECONDARY, anchor="start"))
    for si, s in enumerate(series):
        pts = [(px(xv), py(v)) for xv, v in zip(xs, s.values) if v is not None]
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{yv:.1f}" for i, (x, yv) in enumerate(pts))
        body.append(f'<path d="{d}" fill="none" stroke="{PALETTE[si]}" stroke-width="2"/>')
        for (x, yv), xv, v in zip(pts, xs, s.values):
            body.append(f'<circle cx="{x:.1f}" cy="{yv:.1f}" r="4" fill="{PALETTE[si]}" stroke="{SURFACE}" '
                        f'stroke-width="2"><title>{escape(f"{s.name} · {xv:g}: {fmt(v)}")}</title></circle>')
    body.append(f'<line x1="{left}" x2="{left + pw}" y1="{py(0):.1f}" y2="{py(0):.1f}" stroke="{AXIS}"/>')
    return _frame(width, height, title, subtitle, body)
