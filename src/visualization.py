"""
Module de visualisation.

Toutes les figures produites par le pipeline sont définies ici :
  - Distribution des classes
  - Distribution de quelques features
  - Matrice de corrélation
  - Importance des features
  - Courbes ROC comparatives
  - Matrices de confusion
  - Comparaison des métriques
"""
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Configuration globale matplotlib/seaborn
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


def _save_figure(path: Path, dpi: int = 150) -> None:
    """Sauvegarde la figure courante et l'affiche."""
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✔ {path.name}")


def plot_class_distribution(
    df: pd.DataFrame,
    label_column: str,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Diagramme en barres + camembert de la distribution des classes."""
    counts = df[label_column].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Barres
    axes[0].bar(counts.index, counts.values, color=["steelblue", "tomato"])
    axes[0].set_title("Distribution des classes (effectifs)")
    axes[0].set_ylabel("Nombre d'échantillons")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom")

    # Camembert
    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=["steelblue", "tomato"], startangle=90)
    axes[1].set_title("Distribution des classes (proportions)")

    _save_figure(output_path, dpi)


def plot_correlation_matrix(
    X: pd.DataFrame,
    output_path: Path,
    n_features: int = 30,
    dpi: int = 150,
) -> None:
    """Heatmap de la matrice de corrélation (top n_features pour lisibilité)."""
    features = X.columns[:n_features]
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        X[features].corr(),
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    plt.title(f"Matrice de corrélation ({n_features} premières features)")
    _save_figure(output_path, dpi)


def plot_feature_importance(
    importances: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
    dpi: int = 150,
) -> None:
    """Barplot horizontal des features les plus importantes."""
    top = importances.head(top_n)
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=top, x="Importance", y="Feature",
        hue="Feature", palette="viridis", legend=False,
    )
    plt.title(f"Top {top_n} features les plus importantes (Random Forest)")
    plt.xlabel("Importance relative")
    _save_figure(output_path, dpi)


def plot_roc_curves(
    roc_data: List[Tuple[str, np.ndarray]],
    y_true: np.ndarray,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """
    Courbes ROC superposées pour comparer les modèles.

    Parameters
    ----------
    roc_data : list of (name, y_score)
        Pour chaque modèle : son nom et ses scores de probabilité.
    y_true : np.ndarray
        Labels réels.
    output_path : Path
        Chemin de sortie de la figure.
    """
    plt.figure(figsize=(9, 7))

    colors = ["steelblue", "orange", "green", "tomato", "purple", "brown"]
    for (name, y_score), color in zip(roc_data, colors):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=color, linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Hasard (AUC = 0.5)")
    plt.xlabel("Taux de faux positifs (FPR)")
    plt.ylabel("Taux de vrais positifs (TPR)")
    plt.title("Courbes ROC — Comparaison des modèles")
    plt.legend(loc="lower right")
    _save_figure(output_path, dpi)


def plot_confusion_matrices(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """
    Matrices de confusion en grille pour comparer les modèles.

    Parameters
    ----------
    predictions : dict {name: y_pred}
        Prédictions de chaque modèle.
    y_true : np.ndarray
        Labels réels.
    output_path : Path
        Chemin de sortie.
    """
    n_models = len(predictions)
    n_cols = 2
    n_rows = (n_models + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 5.5 * n_rows))
    axes = np.atleast_2d(axes)

    for ax, (name, y_pred) in zip(axes.flatten(), predictions.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["BENIGN", "ATTACK"],
            yticklabels=["BENIGN", "ATTACK"],
            cbar=False,
        )
        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Vérité terrain")
        ax.set_title(name)

    # Cacher les axes vides éventuels
    for ax in axes.flatten()[n_models:]:
        ax.axis("off")

    plt.suptitle("Matrices de confusion — Comparaison des modèles", fontsize=14, y=1.00)
    _save_figure(output_path, dpi)


def plot_metrics_comparison(
    results: pd.DataFrame,
    output_path: Path,
    metrics: list = None,
    dpi: int = 150,
) -> None:
    """Barplot groupé pour comparer les métriques entre modèles."""
    if metrics is None:
        metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

    results_melted = results.melt(
        id_vars="Modèle",
        value_vars=metrics,
        var_name="Métrique",
        value_name="Score",
    )

    plt.figure(figsize=(13, 6))
    sns.barplot(data=results_melted, x="Métrique", y="Score", hue="Modèle", palette="Set2")
    plt.title("Comparaison des métriques par modèle")

    # Zoom adapté au range des scores
    min_score = results_melted["Score"].min()
    if min_score > 0.9:
        plt.ylim(min_score - 0.02, 1.005)
    else:
        plt.ylim(0, 1.05)

    plt.legend(loc="lower right")
    _save_figure(output_path, dpi)
