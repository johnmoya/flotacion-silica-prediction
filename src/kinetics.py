"""Cinética de flotación batch: modelos, ajuste con IC y datos SINTÉTICOS."""
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def first_order(t, r_inf, k):
    """Primer orden clásico: R(t) = R∞ (1 − e^(−k t))."""
    return r_inf * (1 - np.exp(-k * t))


def klimpel(t, r_inf, k):
    """Klimpel (distribución rectangular de constantes entre 0 y k):
    R(t) = R∞ [1 − (1 − e^(−k t)) / (k t)]."""
    t = np.asarray(t, dtype=float)
    return r_inf * (1 - (1 - np.exp(-k * t)) / (k * t))


MODELS = {"Primer orden": first_order, "Klimpel": klimpel}

# Bounds físicos: 0 ≤ R∞ ≤ 100 %, k > 0 (min⁻¹)
BOUNDS = ([0.0, 1e-6], [100.0, 50.0])


def synthetic_batch_test(times, r_inf=90.0, k=1.5, noise_sd=1.5, model=klimpel, seed=0):
    """Prueba batch SINTÉTICA: recuperación acumulada (%) generada con `model` + ruido normal.

    No son datos de laboratorio. Sirven para mostrar el procedimiento de ajuste
    y verificar que recupera parámetros conocidos.
    """
    rng = np.random.default_rng(seed)
    clean = model(np.asarray(times, dtype=float), r_inf, k)
    return np.clip(clean + rng.normal(0, noise_sd, len(times)), 0, 100)


def fit_kinetic(model, t, recovery, p0=(80.0, 1.0), bounds=BOUNDS, level=0.95) -> dict:
    """Ajuste por mínimos cuadrados no lineales con IC t de Student.

    IC = θ̂ ± t(1−α/2, n−p) · EE, con EE de la matriz de covarianza de curve_fit
    (σ estimado desde los residuos, `absolute_sigma=False`). Con n chico, t de
    Student y no normal.
    """
    t, recovery = np.asarray(t, float), np.asarray(recovery, float)
    params, cov = curve_fit(model, t, recovery, p0=p0, bounds=bounds, absolute_sigma=False)
    n, p = len(t), len(params)
    se = np.sqrt(np.diag(cov))
    t_crit = stats.t.ppf(1 - (1 - level) / 2, df=n - p)
    residuals = recovery - model(t, *params)
    sse = float(np.sum(residuals ** 2))
    aic = n * np.log(sse / n) + 2 * p
    return {
        "params": params,
        "se": se,
        "ci_low": params - t_crit * se,
        "ci_high": params + t_crit * se,
        "residuals": residuals,
        "sse": sse,
        "rmse": float(np.sqrt(sse / (n - p))),
        "aicc": float(aic + 2 * p * (p + 1) / (n - p - 1)),
        "corr_params": float(cov[0, 1] / (se[0] * se[1])),
        "t_crit": float(t_crit),
    }
