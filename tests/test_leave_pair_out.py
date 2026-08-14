from __future__ import annotations

import pytest

from analysis.leave_pair_out import split_contrast


def test_split_contrast_uses_difference_in_relative_improvements() -> None:
    pair_metrics = {
        "fvu_plain": 1.0,
        "fvu_pairwise": 0.8,
        "fvu_ssae_mean": 0.7,
    }
    random_metrics = {
        "fvu_plain": 1.0,
        "fvu_pairwise": 0.6,
        "fvu_ssae_mean": 0.5,
    }

    contrast = split_contrast(pair_metrics, random_metrics)

    assert contrast["h19_gain_contraction"] == pytest.approx(-0.2)
    assert contrast["pairwise_gain_contraction"] == pytest.approx(-0.2)
    assert contrast["pair_gap_closed"] == pytest.approx(2.0 / 3.0)
    assert contrast["random_gap_closed"] == pytest.approx(0.8)
