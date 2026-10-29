"""Fase 4: explicabilidad del mejor modelo de árboles mediante SHAP."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import GroupShuffleSplit

from . import config
from .supervised import NUMERIC_FEATURES, preparar_datos


MODEL_PATH = config.REPORTS_DIR / "models" / "modelo_arbol_xai.joblib"


def _guardar_figura(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def _reproducir_test(df: pd.DataFrame):
    """Reconstruye exactamente el test agrupado utilizado en la Fase 3."""
    X, y, groups, _ = preparar_datos(df)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=config.RANDOM_STATE,
    )
    _, test_idx = next(splitter.split(X, y, groups))
    return X.iloc[test_idx], y.iloc[test_idx], groups.iloc[test_idx]


def _matriz_para_graficos(pipeline, X_test: pd.DataFrame):
    """Conserva valores originales para que los ejes SHAP sean interpretables."""
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = np.asarray(preprocessor.transform(X_test), dtype=float)
    feature_names = list(preprocessor.get_feature_names_out())
    display = pd.DataFrame(transformed.copy(), columns=feature_names, index=X_test.index)

    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    medians = numeric_pipeline.named_steps["imputer"].statistics_
    for column, median in zip(NUMERIC_FEATURES, medians, strict=True):
        raw = pd.to_numeric(X_test[column], errors="coerce").fillna(float(median))
        display[column] = raw.to_numpy()
    return transformed, display, feature_names


def _seleccionar_casos(y_test: pd.Series, probability: np.ndarray) -> tuple[int, int]:
    y_values = y_test.to_numpy()
    high_candidates = np.flatnonzero((y_values == 1) & (probability >= 0.5))
    low_candidates = np.flatnonzero((y_values == 0) & (probability < 0.5))
    if len(high_candidates) == 0:
        high_candidates = np.flatnonzero(y_values == 1)
    if len(low_candidates) == 0:
        low_candidates = np.flatnonzero(y_values == 0)
    high = int(high_candidates[np.argmax(probability[high_candidates])])
    low = int(low_candidates[np.argmin(probability[low_candidates])])
    return high, low


def ejecutar_fase4(df: pd.DataFrame) -> dict[str, object]:
    """Calcula explicaciones globales, dependencias e instancias extremas."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = joblib.load(MODEL_PATH)
    X_test, y_test, groups_test = _reproducir_test(df)
    transformed, display, feature_names = _matriz_para_graficos(pipeline, X_test)
    classifier = pipeline.named_steps["classifier"]
    classifier_name = type(classifier).__name__.replace("Classifier", "")
    probability = pipeline.predict_proba(X_test)[:, 1]

    explainer = shap.TreeExplainer(classifier)
    raw_explanation = explainer(transformed)
    explanation = shap.Explanation(
        values=np.asarray(raw_explanation.values),
        base_values=np.asarray(raw_explanation.base_values),
        data=display.to_numpy(),
        feature_names=feature_names,
    )

    importance = pd.DataFrame({
        "variable": feature_names,
        "shap_abs_medio": np.abs(explanation.values).mean(axis=0),
    }).sort_values("shap_abs_medio", ascending=False, ignore_index=True)
    importance.insert(0, "ranking", np.arange(1, len(importance) + 1))
    importance.to_csv(config.REPORTS_DIR / "shap_importancia_global.csv", index=False)

    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title(f"Determinantes globales del rezago escolar ({classifier_name})")
    _guardar_figura(config.FIGURES_DIR / "shap_summary_beeswarm.png")

    shap.dependence_plot(
        "lengua_originaria",
        explanation.values,
        display,
        interaction_index="p1144_25",
        show=False,
    )
    plt.xlabel("Lengua originaria (1 = sí)")
    plt.ylabel("Impacto SHAP sobre el riesgo")
    plt.title("Interacción: lengua originaria e internet en el hogar")
    plt.gcf().axes[-1].set_ylabel("Internet en el hogar (1 = sí)")
    _guardar_figura(config.FIGURES_DIR / "shap_dependencia_lengua_internet.png")

    shap.dependence_plot(
        "log_gasto_pc",
        explanation.values,
        display,
        interaction_index="escale_ratio_alumno_docente",
        show=False,
    )
    plt.xlabel("Logaritmo del gasto per cápita")
    plt.ylabel("Impacto SHAP sobre el riesgo")
    plt.title("Interacción: gasto familiar y ratio alumno/docente")
    plt.gcf().axes[-1].set_ylabel("Ratio alumno/docente")
    _guardar_figura(config.FIGURES_DIR / "shap_dependencia_gasto_ratio.png")

    high, low = _seleccionar_casos(y_test, probability)
    for position, label, title in [
        (high, "alto_riesgo", "Caso de alto riesgo correctamente identificado"),
        (low, "bajo_riesgo", "Caso de bajo riesgo correctamente identificado"),
    ]:
        shap.plots.waterfall(explanation[position], max_display=12, show=False)
        plt.title(
            f"{title}\nP(rezago)={probability[position]:.3f}; salida SHAP en log-odds",
            fontsize=11,
        )
        _guardar_figura(config.FIGURES_DIR / f"shap_waterfall_{label}.png")

    cases = []
    for position, label in [(high, "alto_riesgo"), (low, "bajo_riesgo")]:
        cases.append({
            "caso": label,
            "posicion_test": int(position),
            "indice_original": int(X_test.index[position]),
            "conglome_25": str(groups_test.iloc[position]),
            "rezago_real": int(y_test.iloc[position]),
            "probabilidad_modelo": float(probability[position]),
            "prediccion_umbral_05": int(probability[position] >= 0.5),
        })
    cases_df = pd.DataFrame(cases)
    cases_df.to_csv(config.REPORTS_DIR / "shap_casos_extremos.csv", index=False)

    metadata = {
        "modelo_explicado": type(classifier).__name__,
        "metodo": "shap.TreeExplainer",
        "escala_shap": "log-odds (salida cruda del clasificador)",
        "n_test_explicado": int(len(X_test)),
        "n_variables_transformadas": int(len(feature_names)),
        "top_15": importance.head(15).to_dict(orient="records"),
        "casos_extremos": cases,
        "nota_interpretacion": (
            "SHAP describe asociaciones del modelo, no efectos causales. "
            "Los ejes de las variables numéricas se muestran en su escala original."
        ),
    }
    metadata_path = config.REPORTS_DIR / "shap_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return {
        "importance": importance,
        "cases": cases_df,
        "metadata": metadata,
        "explanation": explanation,
    }


if __name__ == "__main__":
    dataset = pd.read_parquet(config.DATASET_FINAL_PARQUET)
    ejecutar_fase4(dataset)
