"""Construcción del dataset maestro ENAHO-ESCALE 2025."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data_loader import cargar_todas_las_bases


def _numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _unique_households(df: pd.DataFrame, name: str) -> pd.DataFrame:
    duplicated = df.duplicated(config.KEYS_HOGAR).sum()
    if duplicated:
        print(f"[WARN] {name}: {duplicated:,} duplicados de hogar; se conserva la primera fila.")
        df = df.drop_duplicates(config.KEYS_HOGAR, keep="first")
    return df


def _aggregate_escale(padron: pd.DataFrame, padlocal: pd.DataFrame) -> pd.DataFrame:
    padron = padron.copy()
    padron = _numeric(padron, ["TALUM", "TDOC", "TSEC", "NLAT_IE", "NLONG_IE"])
    padron["es_ie_publica"] = padron["GESTION"].eq("1").astype("Int8")
    padron["es_ie_rural"] = (
        padron["DAREACENSO"].str.casefold().eq("rural")
        | padron["AREA_CENSO"].eq("2")
    ).astype("Int8")
    district = padron.groupby(config.KEY_UBIGEO_ESCALE, dropna=False).agg(
        escale_total_alumnos=("TALUM", "sum"),
        escale_total_docentes=("TDOC", "sum"),
        escale_total_secciones=("TSEC", "sum"),
        escale_pct_colegios_publicos=("es_ie_publica", "mean"),
        escale_pct_colegios_rurales=("es_ie_rural", "mean"),
        escale_lat_centroide=("NLAT_IE", "mean"),
        escale_lon_centroide=("NLONG_IE", "mean"),
        escale_total_ies=("GESTION", "size"),
    ).reset_index()
    district["escale_ratio_alumno_docente"] = (
        district["escale_total_alumnos"]
        / district["escale_total_docentes"].replace(0, np.nan)
    )

    local = padlocal.copy()
    local["es_local_rural"] = local["AREA_LOC"].eq("2").astype("Int8")
    local_agg = local.groupby(config.KEY_UBIGEO_ESCALE, dropna=False).agg(
        escale_total_locales=("AREA_LOC", "size"),
        escale_pct_locales_rurales=("es_local_rural", "mean"),
    ).reset_index()
    return district.merge(local_agg, on=config.KEY_UBIGEO_ESCALE, how="left", validate="one_to_one")


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        config.COL_EDAD, config.COL_LENGUA, config.COL_MATRICULA,
        config.COL_ASISTE, config.COL_NIVEL_ACTUAL, config.COL_GRADO_ACTUAL,
        "p1141_25", "p1142_25", "p1144_25", "p314a_25",
        config.COL_GASTO_PC, config.COL_INGRESO_PC, config.COL_POBREZA,
        config.COL_ESTRATO, config.COL_FACTOR,
    ]
    df = _numeric(df, [c for c in numeric_cols if c in df.columns])
    level = df[config.COL_NIVEL_ACTUAL]
    grade = df[config.COL_GRADO_ACTUAL]
    age = df[config.COL_EDAD]
    primary = level.eq(2) & grade.between(1, 6)
    secondary = level.eq(3) & grade.between(1, 5)
    age_norm = pd.Series(np.nan, index=df.index, dtype="float64")
    age_norm.loc[primary] = grade.loc[primary] + 5
    age_norm.loc[secondary] = grade.loc[secondary] + 11
    evaluable = (df[config.COL_ASISTE].eq(1) | df[config.COL_MATRICULA].eq(1)) & age_norm.notna()
    df["edad_normativa"] = age_norm
    df["poblacion_evaluable_rezago"] = evaluable.astype("Int8")
    df[config.TARGET_REZAGO] = pd.Series(pd.NA, index=df.index, dtype="Int8")
    df.loc[evaluable, config.TARGET_REZAGO] = (age.loc[evaluable] > age_norm.loc[evaluable] + config.TOLERANCIA_REZAGO).astype("Int8")

    originaria_codes = {1, 2, 3, 10, 11, 12, 13, 14, 15}
    df["lengua_originaria"] = df[config.COL_LENGUA].isin(originaria_codes).astype("Int8")
    attending = df[config.COL_ASISTE].eq(1)
    no_home_internet = df["p1144_25"].ne(1)
    no_recent_use = df["p314a_25"].ne(1)
    no_computer = df["p1141_25"].ne(1) & df["p1142_25"].ne(1)
    df[config.TARGET_EXCLUSION_DIGITAL] = (attending & no_home_internet & no_recent_use & no_computer).astype("Int8")
    df["equipamiento_tic"] = (
        df[["p1141_25", "p1142_25", "p1143_25"]].eq(1).sum(axis=1).astype("Int8")
    )
    df["log_gasto_pc"] = np.log1p(df[config.COL_GASTO_PC].clip(lower=0))
    df["log_ingreso_pc"] = np.log1p(df[config.COL_INGRESO_PC].clip(lower=0))
    return df


def construir_dataset_maestro(bases: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    bases = bases or cargar_todas_las_bases()
    m300 = bases["m300"].copy()
    m200 = bases["m200"].copy()
    m100 = _unique_households(bases["m100"].copy(), "M100")
    sumaria = _unique_households(bases["sumaria"].copy(), "Sumaria")
    before = len(m300)
    m300 = m300[m300[config.KEY_UBIGEO_ENAHO].notna() & m300["conglome_25"].notna()].copy()
    print(f"[INFO] M300 activas 2025: {len(m300):,} de {before:,}")
    persona = m300.merge(m200, on=config.KEYS_PERSONA, how="left", validate="one_to_one", suffixes=("", "_m200"))
    hogar = sumaria.merge(m100, on=config.KEYS_HOGAR, how="left", validate="one_to_one", suffixes=("", "_m100"))
    hogar = hogar.drop(columns=[f"{config.KEY_UBIGEO_ENAHO}_m100"], errors="ignore")
    maestro = persona.merge(hogar, on=config.KEYS_HOGAR, how="left", validate="many_to_one", suffixes=("", "_hogar"))
    escale = _aggregate_escale(bases["padron"], bases["padlocal"])
    maestro = maestro.merge(escale, left_on=config.KEY_UBIGEO_ENAHO, right_on=config.KEY_UBIGEO_ESCALE, how="left", validate="many_to_one")
    maestro = maestro.drop(columns=[config.KEY_UBIGEO_ESCALE], errors="ignore")
    maestro[config.COL_EDAD] = pd.to_numeric(maestro[config.COL_EDAD], errors="coerce")
    before_school_age = len(maestro)
    maestro = maestro[maestro[config.COL_EDAD].between(config.EDAD_MIN, config.EDAD_MAX)].copy()
    print(
        f"[INFO] Población escolar {config.EDAD_MIN}-{config.EDAD_MAX} años: "
        f"{len(maestro):,} de {before_school_age:,} registros"
    )
    maestro = _add_targets(maestro)
    print(f"[INFO] Dataset maestro: {maestro.shape[0]:,} filas x {maestro.shape[1]:,} columnas")
    return maestro


def _dictionary(df: pd.DataFrame) -> pd.DataFrame:
    descriptions = {
        config.TARGET_REZAGO: "Indicador de rezago escolar según edad normativa y grado actual.",
        config.TARGET_EXCLUSION_DIGITAL: "Exclusión digital efectiva entre estudiantes que asisten.",
        "edad_normativa": "Edad normativa calculada según nivel y grado actual.",
        "lengua_originaria": "Indicador de lengua materna originaria.",
        "equipamiento_tic": "Número de dispositivos TIC disponibles en el hogar.",
        "escale_ratio_alumno_docente": "Ratio distrital de alumnos por docente.",
        "escale_total_locales": "Cantidad distrital de locales escolares.",
    }
    rows = []
    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        rango = ""
        if not non_null.empty and pd.api.types.is_numeric_dtype(series):
            rango = f"min={non_null.min()}, max={non_null.max()}"
        rows.append({
            "nombre_columna": column,
            "descripcion": descriptions.get(column, "Variable proveniente de ENAHO, ESCALE o derivada."),
            "fuente": "Derivada" if column in descriptions else ("ESCALE" if column.startswith("escale_") else "ENAHO 2025"),
            "tipo_dato": str(series.dtype),
            "rango_valores": rango,
        })
    return pd.DataFrame(rows)


def exportar_dataset_maestro(df: pd.DataFrame) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.DATASET_FINAL_PARQUET, index=False)
    df.to_csv(config.DATASET_FINAL_CSV, index=False, encoding="utf-8-sig")
    _dictionary(df).to_csv(config.DICCIONARIO_VARIABLES, index=False, encoding="utf-8-sig")
    print(f"[OK] Exportado: {config.DATASET_FINAL_PARQUET}")
    print(f"[OK] Exportado: {config.DATASET_FINAL_CSV}")
    print(f"[OK] Exportado: {config.DICCIONARIO_VARIABLES}")


if __name__ == "__main__":
    exportar_dataset_maestro(construir_dataset_maestro())

