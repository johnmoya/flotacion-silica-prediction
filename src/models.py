"""Modelos: Ridge con escalado y XGBoost con early stopping temporal."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBRegressor

from src.config import SEED

XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 100,
    "eval_metric": "mae",
    "random_state": SEED,
}


def make_ridge(alpha: float = 1.0, clip_sigma: float = 4.0) -> Pipeline:
    """StandardScaler + recorte a ±`clip_sigma` + Ridge.

    El escalador va dentro del Pipeline, así que se ajusta solo con datos de
    entrenamiento. El recorte evita que un sensor fuera del rango visto en
    entrenamiento (p. ej. aire congelado en marzo y luego en otro setpoint)
    se extrapole linealmente.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clip", FunctionTransformer(np.clip, kw_args={"a_min": -clip_sigma, "a_max": clip_sigma})),
        ("ridge", Ridge(alpha=alpha)),
    ])


def fit_xgb_early_stopping(X: pd.DataFrame, y: pd.Series, val_frac: float = 0.2,
                           refit: bool = True, **params) -> XGBRegressor:
    """Elige el número de árboles con early stopping y reentrena con todo `X`.

    El set de early stopping es el último `val_frac` de `X` en el tiempo: el test
    nunca se usa para decidir cuántos árboles entrenar. Sin el reentrenamiento,
    el modelo final perdería justo el tramo más reciente del entrenamiento.
    """
    cut = int(len(X) * (1 - val_frac))
    config = {**XGB_PARAMS, **params}
    model = XGBRegressor(**config)
    model.fit(X.iloc[:cut], y.iloc[:cut],
              eval_set=[(X.iloc[cut:], y.iloc[cut:])], verbose=False)
    if not refit:
        return model
    final = {k: v for k, v in config.items() if k != "early_stopping_rounds"}
    final["n_estimators"] = model.best_iteration + 1
    return XGBRegressor(**final).fit(X, y)
