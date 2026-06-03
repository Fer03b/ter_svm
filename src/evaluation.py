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
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    training_time: float = None,
) -> dict:
    """
    Calcule toutes les métriques d'évaluation pour un modèle.

    Parameters
    ----------
    name : str
        Nom du modèle (pour l'identification dans le tableau).
    y_true : np.ndarray
        Labels réels.
    y_pred : np.ndarray
        Prédictions binaires (0/1).
    y_score : np.ndarray
        Scores de probabilité ou decision_function.
    training_time : float, optional
        Temps d'entraînement en secondes.

    Returns
    -------
    dict
        Dictionnaire avec toutes les métriques.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    metrics = {
        "Modèle": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "Taux FP": fpr,
        "ROC-AUC": roc_auc_score(y_true, y_score),
    }

    if training_time is not None:
        metrics["Temps (s)"] = training_time

    return metrics


def print_classification_report(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> None:
    """Affiche le rapport de classification détaillé pour un modèle."""
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=["BENIGN", "ATTACK"]))
    print(f"ROC-AUC : {roc_auc_score(y_true, y_score):.4f}")


def build_comparison_table(results: list) -> pd.DataFrame:
    """
    Construit le tableau de comparaison des modèles.

    Parameters
    ----------
    results : list of dict
        Liste des résultats produits par evaluate_model().

    Returns
    -------
    pd.DataFrame
        Tableau de comparaison.
    """
    df = pd.DataFrame(results)
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
