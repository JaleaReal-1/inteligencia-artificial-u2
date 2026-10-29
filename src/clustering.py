"""Fase 2: clustering y perfilamiento socio-tecnológico."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler

from . import config


CLUSTER_FEATURES = [
    "log_gasto_pc",
    "log_ingreso_pc",
    "internet_hogar",
    "luz_electrica",
    "equipamiento_tic",
    "uso_internet_mes",
    "estrato_rural",
    "escale_ratio_alumno_docente",
    "escale_pct_colegios_publicos",
    "escale_pct_locales_rurales",
]


def preparar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, SimpleImputer, RobustScaler]:
    work = df.copy()
    for column in ["p1144_25", "p1121_25", "p314a_25", "p1141_25", "p1142_25", "p1143_25"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["internet_hogar"] = work["p1144_25"].eq(1).astype(float)
    work["luz_electrica"] = work["p1121_25"].eq(1).astype(float)
    work["uso_internet_mes"] = work["p314a_25"].eq(1).astype(float)
    work["estrato_rural"] = pd.to_numeric(work["estrato_25"], errors="coerce")
    matrix = work[CLUSTER_FEATURES].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X = scaler.fit_transform(imputer.fit_transform(matrix))
    return work, X, imputer, scaler


def evaluar_k(X: np.ndarray, k_values=range(2, 9), random_state: int = 42) -> pd.DataFrame:
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)
        rows.append({
            "k": k,
            "inercia": model.inertia_,
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
        })
    result = pd.DataFrame(rows)
    result["score_balance"] = (
        result["silhouette"].rank(pct=True)
        + (-result["davies_bouldin"]).rank(pct=True)
        + result["calinski_harabasz"].rank(pct=True)
    )
    return result


def seleccionar_k(metrics: pd.DataFrame) -> int:
    return int(metrics.loc[metrics["score_balance"].idxmax(), "k"])


def ajustar_modelos(X: np.ndarray, k: int, random_state: int = 42) -> tuple[KMeans, GaussianMixture]:
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20).fit(X)
    gmm = GaussianMixture(n_components=k, random_state=random_state, covariance_type="full").fit(X)
    return kmeans, gmm


def graficar_metricas(metrics: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plots = [("inercia", "Inercia (WCSS)"), ("silhouette", "Silhouette"),
             ("davies_bouldin", "Davies-Bouldin"), ("calinski_harabasz", "Calinski-Harabasz")]
    for ax, (column, title) in zip(axes.flat, plots):
        sns.lineplot(data=metrics, x="k", y=column, marker="o", ax=ax, color="#0077b6")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def graficar_pca(X: np.ndarray, labels: np.ndarray, output: Path, random_state: int = 42) -> tuple[PCA, np.ndarray]:
    pca = PCA(n_components=3, random_state=random_state)
    coords = pca.fit_transform(X)
    sample = np.arange(len(coords))
    if len(sample) > 20000:
        rng = np.random.default_rng(random_state)
        sample = rng.choice(sample, size=20000, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(x=coords[sample, 0], y=coords[sample, 1], hue=labels[sample], palette="tab10", s=12, alpha=.55, legend="full", ax=axes[0])
    axes[0].set_title("PCA 2D - perfiles socio-tecnológicos")
    sns.scatterplot(x=coords[sample, 0], y=coords[sample, 2], hue=labels[sample], palette="tab10", s=12, alpha=.55, legend=False, ax=axes[1])
    axes[1].set_title("PCA componentes 1 y 3")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pca, coords


def graficar_umap(X: np.ndarray, labels: np.ndarray, output: Path, random_state: int = 42) -> object:
    import umap

    sample = np.arange(len(X))
    if len(sample) > 500:
        rng = np.random.default_rng(random_state)
        sample = rng.choice(sample, size=500, replace=False)
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        n_epochs=75,
        low_memory=True,
        random_state=random_state,
        init="random",
        n_jobs=1,
    )
    coords = reducer.fit_transform(X[sample])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels[sample], palette="tab10", s=12, alpha=.55, ax=ax)
    ax.set_title("UMAP 2D - perfiles socio-tecnológicos")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return reducer


def crear_perfiles(work: pd.DataFrame, labels: np.ndarray, k: int) -> pd.DataFrame:
    work = work.copy()
    work["cluster_kmeans"] = labels
    profile_cols = CLUSTER_FEATURES + ["rezago_escolar", "exclusion_digital"]
    profiles = work.groupby("cluster_kmeans")[profile_cols].mean(numeric_only=True).reset_index()
    profiles["n"] = work.groupby("cluster_kmeans").size().values
    profiles["tasa_rezago"] = work.groupby("cluster_kmeans")["rezago_escolar"].mean().values
    profiles["tasa_exclusion_digital"] = work.groupby("cluster_kmeans")["exclusion_digital"].mean().values
    profiles["nombre_cluster"] = "Perfil socio-tecnológico mixto"
    rural_idx = profiles["estrato_rural"].idxmax()
    urban_idx = profiles["estrato_rural"].idxmin()
    profiles.loc[rural_idx, "nombre_cluster"] = "Rural vulnerable con aislamiento digital y dependencia pública"
    profiles.loc[urban_idx, "nombre_cluster"] = "Urbano consolidado con alta conectividad y mejores recursos"
    return profiles


def ejecutar_fase2(df: pd.DataFrame, output_dir: Path | None = None, random_state: int = 42) -> dict[str, object]:
    output_dir = output_dir or (config.REPORTS_DIR / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    work, X, imputer, scaler = preparar_features(df)
    # Las métricas de silueta son O(n²); 5,000 casos dan una muestra estable y reproducible.
    sample_size = min(len(X), 3000)
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(len(X), size=sample_size, replace=False)
    print(f"[INFO] Evaluando k=2..8 sobre {sample_size:,} observaciones...", flush=True)
    metrics = evaluar_k(X[sample_idx], random_state=random_state)
    k = seleccionar_k(metrics)
    print(f"[INFO] k óptimo seleccionado: {k}", flush=True)
    kmeans, gmm = ajustar_modelos(X[sample_idx], k, random_state=random_state)
    labels_km_sample = kmeans.predict(X[sample_idx])
    labels_gmm_sample = gmm.predict(X[sample_idx])
    comparison = pd.DataFrame([
        {
            "modelo": "KMeans",
            "silhouette": silhouette_score(X[sample_idx], labels_km_sample),
            "davies_bouldin": davies_bouldin_score(X[sample_idx], labels_km_sample),
            "calinski_harabasz": calinski_harabasz_score(X[sample_idx], labels_km_sample),
        },
        {
            "modelo": "GaussianMixture",
            "silhouette": silhouette_score(X[sample_idx], labels_gmm_sample),
            "davies_bouldin": davies_bouldin_score(X[sample_idx], labels_gmm_sample),
            "calinski_harabasz": calinski_harabasz_score(X[sample_idx], labels_gmm_sample),
        },
    ])
    labels = kmeans.predict(X)
    profiles = crear_perfiles(work, labels, k)
    metrics.to_csv(config.REPORTS_DIR / "clustering_metricas_k.csv", index=False)
    comparison.to_csv(config.REPORTS_DIR / "clustering_comparacion_modelos.csv", index=False)
    profiles.to_csv(config.REPORTS_DIR / "perfiles_clusters.csv", index=False)
    clustered = df.copy()
    clustered["cluster_kmeans"] = labels
    clustered.to_parquet(config.PROCESSED_DIR / "dataset_enaho_escale_2025_clusters.parquet", index=False)
    graficar_metricas(metrics, output_dir / "clustering_metricas_k.png")
    pca, coords = graficar_pca(X, labels, output_dir / "pca_clusters.png", random_state)
    print("[INFO] Ejecutando UMAP sobre muestra reproducible de hasta 500 observaciones...", flush=True)
    graficar_umap(X, labels, output_dir / "umap_clusters.png", random_state)
    print("[INFO] UMAP completado.", flush=True)
    metadata = {
        "k_optimo": k,
        "n_observaciones": len(df),
        "n_features_clustering": len(CLUSTER_FEATURES),
        "varianza_pca_3": float(pca.explained_variance_ratio_.sum()),
        "modelo_comparado": "KMeans vs GaussianMixture",
        "umap": "ejecutado sobre muestra reproducible de hasta 500 observaciones",
    }
    (config.REPORTS_DIR / "clustering_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(profiles.to_string(index=False))
    return {"metrics": metrics, "comparison": comparison, "profiles": profiles, "labels": labels, "pca": pca, "coords": coords, "gmm": gmm}


if __name__ == "__main__":
    dataset = pd.read_parquet(config.DATASET_FINAL_PARQUET)
    ejecutar_fase2(dataset)

