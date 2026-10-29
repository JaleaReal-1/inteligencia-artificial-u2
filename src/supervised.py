"""Fase 3: clasificación supervisada de rezago escolar sin fuga de datos."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config
from .evaluation import (
    calcular_metricas,
    graficar_calibracion,
    graficar_curvas,
    graficar_matriz_confusion,
    resumen_cv,
)


NUMERIC_FEATURES = [
    "log_gasto_pc", "log_ingreso_pc", "lengua_originaria", "equipamiento_tic",
    "p1121_25", "p1141_25", "p1142_25", "p1143_25", "p1144_25", "p314a_25",
    "escale_ratio_alumno_docente", "escale_pct_colegios_publicos",
    "escale_pct_colegios_rurales", "escale_total_locales",
    "escale_pct_locales_rurales",
]

CATEGORICAL_FEATURES = ["p207_25", "p300a_25", "dominio_25", "estrato_25"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def preparar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    data = df[df[config.TARGET_REZAGO].notna()].copy()
    X = data[FEATURES].copy()
    for column in FEATURES:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    y = data[config.TARGET_REZAGO].astype(int)
    groups = data[config.GRUPO_CONGLOME].astype("string")
    weights = pd.to_numeric(data[config.COL_FACTOR], errors="coerce").fillna(1.0)
    weights = weights / weights.mean()
    return X, y, groups, weights


def crear_preprocesador() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)


def crear_modelos(positive_ratio: float) -> dict[str, object]:
    return {
        "Regresion Logistica": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=config.RANDOM_STATE),
        "Arbol CART": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=config.RANDOM_STATE),
        "LightGBM": LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=24, class_weight="balanced", verbosity=-1, random_state=config.RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=250, learning_rate=0.04, max_depth=4, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=positive_ratio, eval_metric="logloss", random_state=config.RANDOM_STATE, n_jobs=-1),
        "CatBoost": CatBoostClassifier(iterations=250, depth=6, learning_rate=0.04, auto_class_weights="Balanced", verbose=False, random_seed=config.RANDOM_STATE, thread_count=-1),
    }


def _fit_pipeline(model, X, y, weights) -> Pipeline:
    pipeline = Pipeline([("preprocessor", crear_preprocesador()), ("classifier", model)])
    pipeline.fit(X, y, classifier__sample_weight=np.asarray(weights))
    return pipeline


def ejecutar_fase3(df: pd.DataFrame) -> dict[str, object]:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = config.REPORTS_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups, weights = preparar_datos(df)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(outer.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train, g_test = groups.iloc[train_idx], groups.iloc[test_idx]
    w_train = weights.iloc[train_idx]
    assert set(g_train).isdisjoint(set(g_test)), "Fuga detectada: conglomerados compartidos."

    n_pos = int(y_train.sum())
    positive_ratio = float((len(y_train) - n_pos) / max(n_pos, 1))
    models = crear_modelos(positive_ratio)
    fold_rows: list[dict[str, object]] = []
    cv = GroupKFold(n_splits=5)
    for name, model in models.items():
        print(f"[CV] {name}", flush=True)
        for fold, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train, g_train), start=1):
            fitted = _fit_pipeline(clone(model), X_train.iloc[fit_idx], y_train.iloc[fit_idx], w_train.iloc[fit_idx])
            prob = fitted.predict_proba(X_train.iloc[val_idx])[:, 1]
            fold_rows.append({"modelo": name, "fold": fold, **calcular_metricas(y_train.iloc[val_idx], prob)})

    fold_metrics = pd.DataFrame(fold_rows)
    cv_summary = resumen_cv(fold_metrics).sort_values("pr_auc_media", ascending=False)
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    test_rows: list[dict[str, object]] = []
    fitted_models: dict[str, Pipeline] = {}
    for name, model in models.items():
        print(f"[TEST] {name}", flush=True)
        fitted = _fit_pipeline(clone(model), X_train, y_train, w_train)
        prob = fitted.predict_proba(X_test)[:, 1]
        predictions[name] = (y_test.to_numpy(), prob)
        test_rows.append({"modelo": name, **calcular_metricas(y_test, prob)})
        fitted_models[name] = fitted

    test_metrics = pd.DataFrame(test_rows).sort_values("pr_auc", ascending=False)
    champion = str(cv_summary.iloc[0]["modelo"])
    champion_model = fitted_models[champion]
    champion_prob = predictions[champion][1]
    tree_names = ["Arbol CART", "Random Forest", "LightGBM", "XGBoost", "CatBoost"]
    tree_summary = cv_summary[cv_summary["modelo"].isin(tree_names)]
    xai_tree = str(tree_summary.iloc[0]["modelo"])

    fold_metrics.to_csv(config.REPORTS_DIR / "modelos_cv_folds.csv", index=False)
    cv_summary.to_csv(config.REPORTS_DIR / "modelos_cv_resumen.csv", index=False)
    test_metrics.to_csv(config.REPORTS_DIR / "modelos_metricas_test.csv", index=False)
    pd.DataFrame({
        "indice_original": y_test.index,
        "y_real": y_test.to_numpy(),
        "probabilidad_rezago": champion_prob,
        "prediccion": (champion_prob >= 0.5).astype(int),
        "conglome_25": g_test.to_numpy(),
    }).to_csv(config.REPORTS_DIR / "predicciones_modelo_campeon.csv", index=False)
    joblib.dump(champion_model, model_dir / "modelo_campeon_rezago.joblib")
    joblib.dump(fitted_models[xai_tree], model_dir / "modelo_arbol_xai.joblib")
    graficar_curvas(predictions, config.FIGURES_DIR)
    graficar_calibracion(predictions, config.FIGURES_DIR)
    graficar_matriz_confusion(y_test, champion_prob, champion, config.FIGURES_DIR)

    metadata = {
        "modelo_campeon": champion,
        "mejor_modelo_arbol_para_xai": xai_tree,
        "criterio_seleccion": "PR-AUC media en GroupKFold de 5 folds sobre entrenamiento",
        "n_total_evaluable": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positivos_total": int(y.sum()),
        "prevalencia": float(y.mean()),
        "conglomerados_train": int(g_train.nunique()),
        "conglomerados_test": int(g_test.nunique()),
        "conglomerados_compartidos": 0,
        "features": FEATURES,
        "variables_excluidas_anti_leakage": ["p208a_25", "p308a_25", "p308b_25", "edad_normativa", "poblacion_evaluable_rezago"],
    }
    (config.REPORTS_DIR / "modelos_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    print(test_metrics.to_string(index=False), flush=True)
    return {
        "cv_folds": fold_metrics,
        "cv_summary": cv_summary,
        "test_metrics": test_metrics,
        "champion": champion,
        "champion_model": champion_model,
        "predictions": predictions,
    }


if __name__ == "__main__":
    dataset = pd.read_parquet(config.DATASET_FINAL_PARQUET)
    ejecutar_fase3(dataset)

