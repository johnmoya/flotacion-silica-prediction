"""Ingeniería de variables a resolución horaria.

Supuesto de operación: al cierre de la hora t queremos estimar la sílice del
ensayo de la hora t. Disponemos del proceso hasta t, pero los resultados de
laboratorio (sílice del concentrado y leyes de alimentación) llegan con
`lab_delay` horas de retraso.
"""
import pandas as pd

from src.config import LEAKY_COLUMNS, TARGET
from src.data import FEED_COLUMNS

PROCESS_COLUMNS = (
    ["Starch Flow", "Amina Flow", "Ore Pulp Flow", "Ore Pulp pH", "Ore Pulp Density"]
    + [f"Flotation Column {i:02d} Air Flow" for i in range(1, 8)]
    + [f"Flotation Column {i:02d} Level" for i in range(1, 8)]
)
# Horas en que cambian las leyes de alimentación (compósitos por turno, ver 01_eda)
FEED_SHIFT_BOUNDARIES = (1, 7, 16)


def hourly_aggregate(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Media y desviación estándar horaria del proceso sobre una grilla regular.

    - Se eliminan filas duplicadas exactas (historiador congelado).
    - Las horas congeladas quedan en NaN.
    - Leyes de alimentación: un valor por hora (no tienen variación intra-hora).
    - Target: solo en horas con ensayo real; `target_valido` lo indica.
    - `ensayo_conocido`: ensayos como se ven en el momento, para construir features.
      La bandera `target_repetido` mira horas posteriores, así que solo sirve para
      elegir qué horas se evalúan.
    """
    clean = df.drop(columns=LEAKY_COLUMNS).drop_duplicates()
    g = clean.groupby("date")
    hourly = pd.concat([
        g[PROCESS_COLUMNS].mean(),
        g[PROCESS_COLUMNS].std().add_suffix("_std"),
        g[FEED_COLUMNS].first(),
        g[TARGET].mean(),
    ], axis=1).asfreq("h")

    flags = flags.reindex(hourly.index)
    frozen = flags["hora_congelada"].fillna(False).astype(bool)
    hourly.loc[frozen] = float("nan")

    valid = ~(flags["target_interpolado"] | flags["target_repetido"]).fillna(True).astype(bool)
    hourly["target_valido"] = valid & ~frozen & hourly[TARGET].notna()
    interpolated = flags["target_interpolado"].fillna(True).astype(bool)
    assay = hourly[TARGET].where(~interpolated)
    hourly["ensayo_conocido"] = assay.where(assay != assay.shift())
    hourly[TARGET] = hourly[TARGET].where(hourly["target_valido"])
    return hourly


def process_lags(hourly: pd.DataFrame, columns: list[str], lags=(1, 2, 3)) -> pd.DataFrame:
    """Rezagos de las medias horarias (tiempo de residencia en las columnas)."""
    return pd.concat(
        [hourly[columns].shift(k).add_suffix(f"_lag{k}") for k in lags], axis=1)


def _shift_end(ts: pd.Timestamp, boundaries=FEED_SHIFT_BOUNDARIES) -> pd.Timestamp:
    """Primer límite de turno estrictamente posterior a `ts`."""
    day = ts.normalize()
    candidates = [day + pd.Timedelta(hours=h) for h in boundaries]
    candidates.append(day + pd.Timedelta(days=1, hours=min(boundaries)))
    return min(c for c in candidates if c > ts)


def available_feed(hourly: pd.DataFrame, lab_delay: int,
                   boundaries=FEED_SHIFT_BOUNDARIES) -> pd.DataFrame:
    """Leyes de alimentación tal como se conocen en cada hora.

    El valor asignado a un turno es un compósito: recién se conoce al cerrar el
    turno más `lab_delay` horas. En cada hora t se usa el último compósito ya
    disponible (nunca el del turno en curso).
    """
    feed = hourly[FEED_COLUMNS].dropna()
    available_from = feed.index.map(lambda ts: _shift_end(ts, boundaries)) + pd.Timedelta(hours=lab_delay)
    composites = (feed.assign(available_from=available_from)
                  .groupby("available_from").first()
                  .reset_index())

    grid = pd.DataFrame({"date": hourly.index})
    known = pd.merge_asof(grid, composites, left_on="date", right_on="available_from",
                          direction="backward", tolerance=pd.Timedelta(hours=48))
    known = known.set_index("date")[FEED_COLUMNS]
    return known.add_suffix("_disponible")


def silica_history(target: pd.Series, lab_delay: int, n_lags: int = 3,
                   window: int = 6, max_stale: int = 24) -> pd.DataFrame:
    """Sílice de ensayos anteriores, con al menos `lab_delay` horas de antigüedad.

    `target` debe tener NaN donde no hubo ensayo nuevo. Cada rezago se rellena
    hacia adelante con el último ensayo conocido (máximo `max_stale` horas),
    como lo vería un operador en la sala de control.
    """
    shifted = target.shift(lab_delay)
    out = pd.DataFrame(index=target.index)
    out["silica_ultimo"] = shifted.ffill(limit=max_stale)
    for k in range(1, n_lags):
        out[f"silica_lag{lab_delay + k}"] = target.shift(lab_delay + k).ffill(limit=max_stale)
    out[f"silica_media{window}h"] = shifted.rolling(window, min_periods=1).mean().ffill(limit=max_stale)

    last_valid_time = pd.Series(target.index.where(target.notna()), index=target.index).shift(lab_delay).ffill()
    out["horas_desde_ensayo"] = (target.index - last_valid_time).dt.total_seconds() / 3600
    return out


def build_features(df: pd.DataFrame, flags: pd.DataFrame, lab_delay: int = 2,
                   lags=(1, 2, 3), feed_as_composite: bool = True):
    """Matriz de features X, target y y máscara de filas utilizables.

    Filas utilizables: hora con ensayo real y sin NaN en ninguna feature.
    """
    if lab_delay < 1:
        raise ValueError("lab_delay debe ser >= 1: la sílice de la hora t no se conoce en t.")

    hourly = hourly_aggregate(df, flags)
    std_cols = [f"{c}_std" for c in PROCESS_COLUMNS]

    if feed_as_composite:
        feed = available_feed(hourly, lab_delay)
    else:
        feed = hourly[FEED_COLUMNS]

    X = pd.concat([
        hourly[PROCESS_COLUMNS],
        hourly[std_cols],
        process_lags(hourly, PROCESS_COLUMNS, lags),
        feed,
        silica_history(hourly["ensayo_conocido"], lab_delay),
    ], axis=1)
    y = hourly[TARGET]
    usable = hourly["target_valido"] & X.notna().all(axis=1)
    return X, y, usable
