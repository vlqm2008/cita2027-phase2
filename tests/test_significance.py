"""Unit tests for statistical significance utilities."""
import numpy as np
import pytest

from cita2027.adaptation.significance import (
    circular_block_bootstrap_ci,
    paired_wilcoxon_test,
    holm_bonferroni_correction,
)


def test_circular_block_bootstrap_ci():
    rng = np.random.RandomState(42)
    # Strictly negative differences (candidate beats reference)
    diff = -0.5 + rng.normal(0, 0.05, size=200)

    res = circular_block_bootstrap_ci(diff, block_lengths=[5, 10, 20], num_bootstrap=200, seed=42)
    assert res["n"] == 200
    assert res["mean_diff"] < 0
    assert res["all_exclude_zero"] is True
    assert "b5" in res["ci"]
    assert res["ci"]["b5"][1] < 0  # Upper bound is negative


def test_wilcoxon_and_holm():
    rng = np.random.RandomState(42)
    cand = rng.normal(1.0, 0.1, size=100)
    ref = rng.normal(1.5, 0.1, size=100)  # cand is clearly better

    stat, pval = paired_wilcoxon_test(cand, ref)
    assert pval < 0.001

    p_vals = [0.001, 0.04, 0.03, 0.5]
    adj_p, rejected = holm_bonferroni_correction(p_vals, alpha=0.05)
    assert len(adj_p) == 4
    assert adj_p[0] <= adj_p[2] <= adj_p[1] <= adj_p[3]
    assert rejected[0] is True
