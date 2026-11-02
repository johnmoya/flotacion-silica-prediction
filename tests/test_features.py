"""Tests de no-fuga: ninguna feature de la hora t puede depender de información
que no estaba disponible en t. Se perturba un dato y se verifica qué filas cambian.

Uso: python -m pytest tests/
"""
import numpy as np
import pandas as pd
import pytest

from src.config import LEAKY_COLUMNS, TARGET
from src.data import FEED_COLUMNS, hourly_quality_flags
from src.features import PROCESS_COLUMNS, _shift_end, build_features

N_HOURS = 96
START = pd.Timestamp("2017-05-01 01:00")


def make_raw(seed: int = 0) -> pd.DataFrame:
    """Datos sintéticos con la misma estructura que el CSV (3 filas por hora)."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(START, periods=N_HOURS, freq="h")
    dates = hours.repeat(3)
    df = pd.DataFrame({"date": dates})
    for col in PROCESS_COLUMNS:
        df[col] = rng.normal(100, 10, len(df))
    feed_block = pd.Series(hours.map(_shift_end), index=hours)
    feed_values = {b: rng.uniform(10, 20) for b in feed_block.unique()}
    df["% Silica Feed"] = feed_block.map(feed_values).reindex(dates).to_numpy()
    df["% Iron Feed"] = 70 - df["% Silica Feed"]
    hourly_target = pd.Series(rng.uniform(1, 5, N_HOURS), index=hours)
    df[TARGET] = hourly_target.reindex(dates).to_numpy()
    df[LEAKY_COLUMNS[0]] = 68 - hourly_target.reindex(dates).to_numpy()
    return df


def features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    X, _, _ = build_features(df, hourly_quality_flags(df), **kwargs)
    return X


def changed_rows(X_a: pd.DataFrame, X_b: pd.DataFrame, cols: list[str]) -> pd.DatetimeIndex:
    diff = ~np.isclose(X_a[cols], X_b[cols], equal_nan=True)
    return X_a.index[diff.any(axis=1)]


@pytest.mark.parametrize("lab_delay", [1, 2, 3])
def test_silica_features_respect_lab_delay(lab_delay):
    df = make_raw()
    t0 = START + pd.Timedelta(hours=40)
    perturbed = df.copy()
    perturbed.loc[perturbed["date"] == t0, TARGET] += 1.0

    X_a, X_b = features(df, lab_delay=lab_delay), features(perturbed, lab_delay=lab_delay)
    silica_cols = [c for c in X_a.columns if c.startswith("silica_")]
    changed = changed_rows(X_a, X_b, silica_cols)
    assert len(changed) > 0
    assert changed.min() >= t0 + pd.Timedelta(hours=lab_delay)


@pytest.mark.parametrize("lab_delay", [1, 2])
def test_feed_composite_only_after_shift_closes(lab_delay):
    df = make_raw()
    t0 = START + pd.Timedelta(hours=30)
    block_end = _shift_end(t0)
    block_hours = df["date"].map(_shift_end) == block_end
    perturbed = df.copy()
    perturbed.loc[block_hours, FEED_COLUMNS] += 1.0

    X_a, X_b = features(df, lab_delay=lab_delay), features(perturbed, lab_delay=lab_delay)
    feed_cols = [c for c in X_a.columns if c.endswith("_disponible")]
    changed = changed_rows(X_a, X_b, feed_cols)
    assert len(changed) > 0
    assert changed.min() >= block_end + pd.Timedelta(hours=lab_delay)


def test_repeat_flag_does_not_leak_into_silica_features():
    # Que un ensayo se repita en las horas siguientes no se sabe en el momento:
    # la validez de ese ensayo como feature no puede depender de ello.
    df = make_raw()
    t0 = START + pd.Timedelta(hours=40)
    value = df.loc[df["date"] == t0, TARGET].iloc[0]
    perturbed = df.copy()
    for k in (1, 2):
        perturbed.loc[perturbed["date"] == t0 + pd.Timedelta(hours=k), TARGET] = value

    X_a, X_b = features(df), features(perturbed)
    silica_cols = [c for c in X_a.columns if c.startswith("silica_")]
    changed = changed_rows(X_a, X_b, silica_cols)
    assert changed.min() >= t0 + pd.Timedelta(hours=1 + 2)


def test_process_features_use_no_future_hours():
    df = make_raw()
    t0 = START + pd.Timedelta(hours=50)
    perturbed = df.copy()
    perturbed.loc[perturbed["date"] == t0, "Amina Flow"] += 50.0

    X_a, X_b = features(df), features(perturbed)
    changed = changed_rows(X_a, X_b, list(X_a.columns))
    assert changed.min() == t0
    assert changed.max() == t0 + pd.Timedelta(hours=3)


def test_leaky_column_and_zero_delay_rejected():
    df = make_raw()
    X = features(df)
    assert not any(LEAKY_COLUMNS[0] in c for c in X.columns)
    assert TARGET not in X.columns
    with pytest.raises(ValueError):
        features(df, lab_delay=0)
