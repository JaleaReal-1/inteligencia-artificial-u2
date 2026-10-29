"""Métricas y visualizaciones para clasificación supervisada."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calcular_metricas(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_positivo": f1_score(y_true, y_pred, pos_label=1),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
    }


def graficar_curvas(predictions: dict[str, tuple[np.ndarray, np.ndarray]], output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, (y_true, y_prob) in predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_true, y_prob):.3f})")
        axes[1].plot(recall, precision, label=f"{name} ({average_precision_score(y_true, y_prob):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(title="Curvas ROC - conjunto de prueba", xlabel="FPR", ylabel="TPR")
    axes[1].set(title="Curvas Precision-Recall - conjunto de prueba", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "modelos_curvas_roc_pr.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def graficar_calibracion(predictions: dict[str, tuple[np.ndarray, np.ndarray]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y_true, y_prob) in predictions.items():
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "--", color="black", label="Calibración perfecta")
    ax.set(title="Curvas de calibración", xlabel="Probabilidad predicha", ylabel="Frecuencia observada")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "modelos_calibracion.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def graficar_matriz_confusion(y_true, y_prob, model_name: str, output_dir: Path) -> None:
    matrix = confusion_matrix(y_true, np.asarray(y_prob) >= 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
    sns.heatmap(matrix / matrix.sum(axis=1, keepdims=True), annot=True, fmt=".2%", cmap="Blues", cbar=False, ax=axes[1])
    axes[0].set_title(f"{model_name}: valores absolutos")
    axes[1].set_title(f"{model_name}: normalizada por clase")
    for ax in axes:
        ax.set(xlabel="Predicción", ylabel="Valor real")
    fig.tight_layout()
    fig.savefig(output_dir / "modelo_campeon_matriz_confusion.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def resumen_cv(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["roc_auc", "pr_auc", "f1_macro", "f1_positivo", "precision", "recall", "brier"]
    mean = fold_metrics.groupby("modelo")[metric_cols].mean().add_suffix("_media")
    std = fold_metrics.groupby("modelo")[metric_cols].std().add_suffix("_std")
    return mean.join(std).reset_index()

