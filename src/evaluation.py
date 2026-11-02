"""Métricas, validación temporal, intervalos y análisis de error."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.config import SEED


def regression_metrics(y_true, y_pred) -> dict:
    """MAE y RMSE en puntos de % SiO2, más R²."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def relative_improvement(model_error: float, baseline_error: float) -> float:
    """Reducción relativa del error respecto al baseline (0.2 = 20 % menos error)."""
    return 1 - model_error / baseline_error


def block_bootstrap_ci(diff: pd.Series, block: str = "48h", reps: int = 4000,
                       level: float = 0.95, seed: int = SEED) -> tuple:
    """IC de la media de `diff` remuestreando bloques de tiempo.

    Los errores horarios están autocorrelacionados: remuestrear horas sueltas
    daría un intervalo demasiado angosto.
    """
    rng = np.random.default_rng(seed)
    blocks = [g.to_numpy() for _, g in diff.groupby(pd.Grouper(freq=block)) if len(g)]
    n = len(blocks)
    means = [np.concatenate([blocks[i] for i in rng.integers(0, n, n)]).mean()
             for _ in range(reps)]
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


def compare_to_baseline(y_true: pd.Series, pred, baseline, **kwargs) -> dict:
    """Diferencia pareada de error absoluto contra un baseline (positiva = gana el modelo)."""
    base_err = (y_true - baseline).abs()
    diff = base_err - (y_true - pred).abs()
    low, high = block_bootstrap_ci(diff, **kwargs)
    return {"dif_MAE": float(diff.mean()), "IC95_inf": low, "IC95_sup": high,
            "mejora": float(diff.mean() / base_err.mean())}


def regime_vs_baseline(y_true: pd.Series, pred: pd.Series, baseline: pd.Series, regime: pd.Series,
                       n_bins: int = 3, labels=("bajo", "medio", "alto")) -> pd.DataFrame:
    """`compare_to_baseline` dentro de cada cuantil de una variable de régimen."""
    bins = pd.qcut(regime.reindex(y_true.index), q=n_bins, labels=labels)
    rows = {}
    for label in labels:
        idx = bins.index[bins == label]
        rows[label] = {"n_horas": len(idx), **compare_to_baseline(y_true[idx], pred[idx], baseline[idx])}
    return pd.DataFrame(rows).T


def error_by_regime(y_true: pd.Series, predictions: dict, regime: pd.Series,
                    n_bins: int = 3, labels=("bajo", "medio", "alto")) -> pd.DataFrame:
    """MAE de cada modelo por cuantiles (terciles por defecto) de una variable de régimen."""
    bins = pd.qcut(regime.reindex(y_true.index), q=n_bins, labels=labels)
    rows = {}
    for name, pred in predictions.items():
        abs_err = (y_true - pred).abs()
        rows[name] = abs_err.groupby(bins, observed=True).mean()
    table = pd.DataFrame(rows)
    table.insert(0, "n_horas", bins.value_counts().reindex(table.index))
    table.insert(1, "rango", regime.reindex(y_true.index).groupby(bins, observed=True)
                 .agg(lambda s: f"{s.min():.1f}–{s.max():.1f}"))
    return table


def interval_report(y_true: pd.Series, lower, upper, nominal: float) -> dict:
    """Cobertura real y ancho medio de un intervalo de predicción."""
    inside = (y_true >= lower) & (y_true <= upper)
    return {"cobertura_nominal": nominal, "cobertura_real": float(np.mean(inside)),
            "ancho_medio": float(np.mean(np.asarray(upper) - np.asarray(lower)))}


def conformal_halfwidth(cal_residuals, alpha: float) -> float:
    """Semiancho del intervalo conformal partido (split conformal) para cobertura 1 - alpha."""
    scores = np.sort(np.abs(np.asarray(cal_residuals)))
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(scores[min(k, n) - 1])


def time_series_cv(fit_predict, X: pd.DataFrame, y: pd.Series, n_splits: int = 5,
                   gap: int = 6) -> pd.DataFrame:
    """MAE por fold con TimeSeriesSplit (cada fold valida en un tramo posterior).

    `fit_predict(X_tr, y_tr, X_val) -> predicciones`. `gap` descarta filas entre
    train y validación para que los rezagos no crucen la frontera.
    """
    rows = []
    for fold, (tr, val) in enumerate(TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X)):
        pred = fit_predict(X.iloc[tr], y.iloc[tr], X.iloc[val])
        rows.append({"fold": fold, "desde": X.index[val[0]], "hasta": X.index[val[-1]],
                     "n_train": len(tr), **regression_metrics(y.iloc[val], pred)})
    return pd.DataFrame(rows)
