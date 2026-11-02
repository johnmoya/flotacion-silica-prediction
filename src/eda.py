"""Utilidades de análisis exploratorio."""
import pandas as pd


def binned_profile(x: pd.Series, y: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Mediana y rango intercuartil de `y` por cuantiles de `x`."""
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    data["bin"] = pd.qcut(data["x"], q=n_bins, duplicates="drop")
    return (data.groupby("bin", observed=True)
            .agg(x=("x", "median"), y_med=("y", "median"),
                 y_q25=("y", lambda s: s.quantile(0.25)),
                 y_q75=("y", lambda s: s.quantile(0.75)), n=("y", "size"))
            .reset_index(drop=True))


def lagged_spearman(y: pd.Series, x: pd.Series, lags: range) -> pd.Series:
    """Correlación de Spearman entre y(t) y x(t + k) para cada k en `lags`.

    k < 0: x antecede a y (x podría influir en y).
    k > 0: x sigue a y (x podría ser una reacción a y).
    Ambas series deben estar sobre la misma grilla horaria regular.
    """
    return pd.Series({k: y.corr(x.shift(-k), method="spearman") for k in lags},
                     name=x.name)
