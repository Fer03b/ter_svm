from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")


def _save_figure(path: Path, dpi: int = 150) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✔ {path.name}")


def plot_prediction_distribution(
    predictions: Dict[str, pd.Series],
    y_true,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Affiche pour chaque modèle la distribution des classes prédites
    et la précision par classe (en pourcentage).
    """
    y_true = pd.Series(y_true)
    classes = sorted(y_true.unique())
    n_models = len(predictions)

    fig, axes = plt.subplots(n_models, 2, figsize=(14, 4 * max(1, n_models)))
    axes = np.atleast_2d(axes)

    for ax_row, (name, y_pred) in zip(axes, predictions.items()):
        y_pred = pd.Series(y_pred)

        # Distribution des classes prédites
        pred_counts = y_pred.value_counts().reindex(classes, fill_value=0)
        pred_perc = (pred_counts / pred_counts.sum()) * 100

        ax = ax_row[0]
        ax.bar(classes, pred_perc, color=sns.color_palette("Set2", len(classes)))
        ax.set_title(f"Distribution prédite — {name}")
        ax.set_ylabel("Pourcentage (%)")
        ax.set_ylim(0, 100)
        for i, v in enumerate(pred_perc.values):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

        # Précision par classe
        per_class_acc = []
        for c in classes:
            mask = y_true == c
            total = mask.sum()
            if total == 0:
                acc = 0.0
            else:
                acc = ((y_pred[mask] == c).sum() / total) * 100
            per_class_acc.append(acc)

        ax2 = ax_row[1]
        ax2.bar(classes, per_class_acc, color=sns.color_palette("muted", len(classes)))
        ax2.set_title(f"Précision par classe (%) — {name}")
        ax2.set_ylabel("Précision (%)")
        ax2.set_ylim(0, 100)
        for i, v in enumerate(per_class_acc):
            ax2.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Distribution des prédictions et précision par classe par modèle", fontsize=14, y=1.02)
    _save_figure(output_path, dpi)
