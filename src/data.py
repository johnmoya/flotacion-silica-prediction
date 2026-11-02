"""Carga del CSV crudo y diagnóstico de frecuencias, huecos y calidad."""
from pathlib import Path

import pandas as pd

from src.config import RAW_CSV, TARGET

FEED_COLUMNS = ["% Iron Feed", "% Silica Feed"]
LAB_COLUMNS = ["% Iron Concentrate", "% Silica Concentrate"]


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Lee el CSV original (coma decimal, `date` con resolución horaria)."""
    return pd.read_csv(path, decimal=",", parse_dates=["date"])


def unique_per_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen de valores únicos por hora de cada columna.

    Una columna que cambia cada ~20 s tiene ~180 únicos/hora; una medición
    horaria o más lenta tiene 1.
    """
    nun = df.groupby("date").nunique()
    return pd.DataFrame({
        "unicos_mediana": nun.median(),
        "unicos_p90": nun.quantile(0.9),
        "unicos_max": nun.max(),
        "pct_horas_con_1_unico": (nun == 1).mean().mul(100).round(1),
    })


def runs(values: pd.Series) -> pd.DataFrame:
    """Agrupa una serie en tramos de valor constante consecutivo.

    Devuelve inicio, fin, largo (n) y valor de cada tramo.
    """
    run_id = (values != values.shift()).cumsum()
    return (values.groupby(run_id)
            .agg(inicio=lambda s: s.index.min(), fin=lambda s: s.index.max(),
                 n="size", valor="first")
            .reset_index(drop=True))


def time_gaps(timestamps: pd.DatetimeIndex, freq: str = "h") -> pd.DataFrame:
    """Tramos de horas faltantes entre el primer y el último timestamp."""
    full = pd.date_range(timestamps.min(), timestamps.max(), freq=freq)
    missing = pd.Series(full.difference(timestamps), dtype="datetime64[us]")
    if missing.empty:
        return pd.DataFrame(columns=["inicio", "fin", "horas"])
    run_id = (missing.diff() != pd.Timedelta(1, freq)).cumsum()
    return (missing.groupby(run_id)
            .agg(inicio="min", fin="max", horas="size")
            .reset_index(drop=True))


def hourly_quality_flags(df: pd.DataFrame, min_ffill_run: int = 3) -> pd.DataFrame:
    """Banderas de calidad por hora.

    - target_interpolado: el target varía dentro de la hora (relleno lineal,
      no es un ensayo real).
    - target_repetido: el valor horario del target se repite >= `min_ffill_run`
      horas seguidas (probable arrastre del último ensayo).
    - hora_congelada: todas las filas de la hora son idénticas (historiador
      congelado).
    """
    g = df.groupby("date")
    flags = pd.DataFrame(index=g.size().index)
    flags["filas"] = g.size()
    flags["target_interpolado"] = g[TARGET].nunique() > 1

    hourly_target = g[TARGET].first().where(~flags["target_interpolado"])
    run_id = (hourly_target != hourly_target.shift()).cumsum()
    run_len = hourly_target.groupby(run_id).transform("size")
    flags["target_repetido"] = hourly_target.notna() & (run_len >= min_ffill_run)

    process_cols = df.columns.drop("date")
    flags["hora_congelada"] = g[process_cols].nunique().max(axis=1) == 1
    return flags


def to_hourly(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Media horaria sobre una grilla horaria regular (huecos = NaN).

    Las horas congeladas quedan en NaN y el target solo se conserva en horas
    con ensayo real (ni interpolado ni repetido). Agrega `target_valido`.
    """
    hourly = df.groupby("date").mean().asfreq("h")
    flags = flags.reindex(hourly.index)
    frozen = flags["hora_congelada"].fillna(False).astype(bool)
    hourly.loc[frozen] = float("nan")

    valid = ~(flags["target_interpolado"] | flags["target_repetido"]).fillna(True).astype(bool)
    hourly["target_valido"] = valid & ~frozen & hourly[TARGET].notna()
    hourly[TARGET] = hourly[TARGET].where(hourly["target_valido"])
    return hourly
