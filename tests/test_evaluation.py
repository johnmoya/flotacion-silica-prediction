import numpy as np
import pandas as pd

from src.evaluation import block_bootstrap_ci, compare_to_baseline


def test_block_bootstrap_contains_mean_and_is_seeded():
    index = pd.date_range("2017-08-01", periods=480, freq="h")
    diff = pd.Series(np.random.default_rng(0).normal(0.1, 1, 480), index=index)
    low, high = block_bootstrap_ci(diff)
    assert low < diff.mean() < high
    assert (low, high) == block_bootstrap_ci(diff)


def test_compare_to_baseline_sign():
    index = pd.date_range("2017-08-01", periods=240, freq="h")
    y = pd.Series(2.0, index=index)
    better = compare_to_baseline(y, y + 0.1, y + 0.5)
    assert better["dif_MAE"] > 0
    assert better["mejora"] == 0.8
