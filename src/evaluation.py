"""
Module d'évaluation des modèles.

Métriques calculées (toutes exigées par le sujet) :
  - Accuracy
  - Precision
  - Recall (rappel)
  - F1-score
  - Taux de faux positifs (FPR)
  - ROC-AUC (en complément)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray = None,
    average: str = "weighted",
    training_time: float = None,
) -> dict:
    """
    Calcule toutes les métriques d'évaluation pour un modèle (multi-classe ou binaire).

    Parameters
    ----------
    y_true : np.ndarray
        Labels réels.
    y_pred : np.ndarray
        Prédictions.
    y_score : np.ndarray, optional
        Scores de probabilité ou decision_function.
    average : str, optional
        Type de moyenning pour multi-classe ("weighted", "macro", "micro").
    training_time : float, optional
        Temps d'entraînement en secondes.

    Returns
    -------
    dict
        Dictionnaire avec toutes les métriques.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    if training_time is not None:
        metrics["Temps (s)"] = training_time

    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Affiche le rapport de classification détaillé."""
    print(classification_report(y_true, y_pred, zero_division=0))


def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    Construit le tableau de comparaison des modèles.

    Parameters
    ----------
    results : dict
        Dictionnaire où les clés sont les noms des modèles 
        et les valeurs sont les dictionnaires de métriques.

    Returns
    -------
    pd.DataFrame
        Tableau de comparaison.
    """
    rows = []
    for model_name, metrics in results.items():
        row = {"Modèle": model_name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def print_comparison_table(df: pd.DataFrame, decimals: int = 4) -> None:
    """Affiche le tableau de comparaison de manière lisible."""
    df_display = df.copy()
    score_cols = ["Accuracy", "Precision", "Recall", "F1-score", "Taux FP", "ROC-AUC"]
    for col in score_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].round(decimals)
    if "Temps (s)" in df_display.columns:
        df_display["Temps (s)"] = df_display["Temps (s)"].round(2)

    print("\n" + "=" * 100)
    print("  COMPARAISON DES MODÈLES")
    print("=" * 100)
    print(df_display.to_string(index=False))


def identify_best_model(df: pd.DataFrame, metric: str = "F1-score") -> str:
    """
    Identifie le meilleur modèle selon une métrique donnée.

    Parameters
    ----------
    df : pd.DataFrame
        Tableau de comparaison.
    metric : str
        Métrique à utiliser pour le classement (par défaut F1-score).

    Returns
    -------
    str
        Nom du meilleur modèle.
    """
    best_idx = df[metric].idxmax()
    best_name = df.loc[best_idx, "Modèle"]
    best_score = df.loc[best_idx, metric]
    print(f"\n→ Meilleur modèle selon {metric} : {best_name} ({best_score:.4f})")
    return best_name
