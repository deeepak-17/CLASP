"""Tests for utils/svg_charts.py — report figures are well-formed and honest about zero."""

from __future__ import annotations

import xml.dom.minidom

import pytest

from evaluation.utils.svg_charts import PALETTE, Series, grouped_bars, lines


def _parse(svg: str) -> xml.dom.minidom.Document:
    return xml.dom.minidom.parseString(svg)


def test_grouped_bars_is_valid_svg_with_one_path_per_value() -> None:
    svg = grouped_bars(["a", "b"], [Series("r1", [0.1, -0.2]), Series("r2", [0.3, None])], title="t & <x>")
    doc = _parse(svg)
    assert len(doc.getElementsByTagName("path")) == 3  # the None value is omitted, not drawn as zero
    assert "t &amp; &lt;x&gt;" in svg


def test_signed_bars_straddle_the_zero_line() -> None:
    svg = grouped_bars(["a"], [Series("s", [-0.5]), Series("t", [0.5])], title="t")
    labels = [t.firstChild.data for t in _parse(svg).getElementsByTagName("text") if t.firstChild]
    assert "0" in labels and any(lbl.startswith("-") for lbl in labels)


def test_lines_draw_a_marker_per_point_and_the_reference() -> None:
    svg = lines([1, 10, 100], [Series("a", [3.0, 2.0, 1.0])], title="t", reference=(2.0, "ref"), log_x=True)
    doc = _parse(svg)
    assert len(doc.getElementsByTagName("circle")) == 3
    assert any(t.firstChild and t.firstChild.data == "ref" for t in doc.getElementsByTagName("text"))


def test_series_beyond_the_palette_are_refused() -> None:
    with pytest.raises(ValueError):
        grouped_bars(["a"], [Series(str(i), [1.0]) for i in range(len(PALETTE) + 1)], title="t")
