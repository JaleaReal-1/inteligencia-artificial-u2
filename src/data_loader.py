"""Carga segura de microdatos ENAHO y ESCALE para el corte 2025."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config


def _read_csv(path: Path, usecols: Iterable[str]) -> pd.DataFrame:
    """Lee un CSV INEI/MINEDU preservando claves y tolerando codificaciones."""
    usecols = list(dict.fromkeys(usecols))
    dtype = {column: str for column in usecols}
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(
                path,
                usecols=usecols,
                dtype=dtype,
                encoding=encoding,
                na_values=["", " ", "  ", "   ", "NA", "N/A"],
                keep_default_na=True,
                low_memory=False,
            )
            df.columns = df.columns.str.replace("\\ufeff", "", regex=False).str.strip()
            return _clean_strings(df)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"No se pudo leer {path}") from last_error


def _clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte espacios en blanco residuales en nulos, sin castear números."""
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()
        df[column] = df[column].replace(r"^$", pd.NA, regex=True)
    return df


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    for column in config.STR_KEYS:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()
    for column in (config.KEY_UBIGEO_ENAHO, config.KEY_UBIGEO_ESCALE):
        if column in df.columns:
            df[column] = df[column].str.zfill(6)
    return df


def cargar_m100() -> pd.DataFrame:
    columns = config.KEYS_HOGAR + [
        config.KEY_UBIGEO_ENAHO,
        "p1121_25", "p1141_25", "p1142_25", "p1143_25", "p1144_25",
    ]
    return _normalize_keys(_read_csv(config.ENAHO_M100, columns))


def cargar_m200() -> pd.DataFrame:
    columns = config.KEYS_PERSONA + ["p207_25", "p203_25", config.COL_EDAD]
    return _normalize_keys(_read_csv(config.ENAHO_M200, columns))


def cargar_m300() -> pd.DataFrame:
    columns = config.KEYS_PERSONA + [
        config.KEY_UBIGEO_ENAHO, config.COL_LENGUA, config.COL_MATRICULA,
        config.COL_ASISTE, config.COL_NIVEL_ACTUAL, config.COL_GRADO_ACTUAL,
        "p314a_25", "p314b_1_25", "p316_1_25", "p316_2_25", "p316_3_25",
        config.COL_FACTOR,
    ]
    return _normalize_keys(_read_csv(config.ENAHO_M300, columns))


def cargar_sumaria() -> pd.DataFrame:
    columns = config.KEYS_HOGAR + [
        config.KEY_UBIGEO_ENAHO, config.COL_GASTO_PC, config.COL_INGRESO_PC,
        config.COL_POBREZA, config.COL_ESTRATO, config.COL_DOMINIO,
        config.COL_FACTOR,
    ]
    return _normalize_keys(_read_csv(config.ENAHO_SUMARIA, columns))


def cargar_padron_escale() -> pd.DataFrame:
    columns = [
        "CODGEO", "GESTION", "DAREACENSO", "AREA_CENSO",
        "TALUM", "TDOC", "TSEC", "NLAT_IE", "NLONG_IE",
    ]
    df = _read_csv(config.ESCALE_PADRON, columns).rename(columns={"CODGEO": config.KEY_UBIGEO_ESCALE})
    return _normalize_keys(df)


def cargar_padlocal_escale() -> pd.DataFrame:
    columns = ["CODGEO", "AREA_LOC", "GES_LOC"]
    df = _read_csv(config.ESCALE_PADLOCAL, columns).rename(columns={"CODGEO": config.KEY_UBIGEO_ESCALE})
    return _normalize_keys(df)


def cargar_todas_las_bases() -> dict[str, pd.DataFrame]:
    """Carga las seis fuentes con las columnas necesarias para la Fase 1."""
    return {
        "m100": cargar_m100(),
        "m200": cargar_m200(),
        "m300": cargar_m300(),
        "sumaria": cargar_sumaria(),
        "padron": cargar_padron_escale(),
        "padlocal": cargar_padlocal_escale(),
    }

