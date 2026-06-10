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
import os
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


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
    training_time: float = None,
) -> dict:
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    if training_time is not None:
        metrics["Temps (s)"] = training_time

    return metrics


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray = None,
    average="weighted",
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
    average : str or tuple/list, optional
        Type de moyenning pour multi-classe ("weighted", "macro", "micro")
        ou un ensemble d'averages pour calculer plusieurs versions.
    training_time : float, optional
        Temps d'entraînement en secondes.

    Returns
    -------
    dict
        Dictionnaire avec toutes les métriques ou
        un dictionnaire de dictionnaires si plusieurs averages sont demandés.
    """
    if isinstance(average, (list, tuple)):
        return {
            avg: _compute_metrics(y_true, y_pred, average=avg, training_time=training_time)
            for avg in average
        }

    return _compute_metrics(y_true, y_pred, average=average, training_time=training_time)


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_division: int = 0,
    digits: int = 4,
) -> None:
    """Affiche le rapport de classification détaillé."""
    print(classification_report(
        y_true,
        y_pred,
        zero_division=zero_division,
        digits=digits,
    ))


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


def print_per_class_prediction_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Affiche un résumé texte par classe des prédictions et des erreurs."""
    y_true = pd.Series(y_true, name="Actual")
    y_pred = pd.Series(y_pred, name="Predicted")
    labels = sorted(pd.unique(pd.concat([y_true, y_pred])))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    print("\n=== Résumé des prédictions par classe ===")
    for actual in labels:
        total = int(df_cm.loc[actual].sum())
        correct = int(df_cm.loc[actual, actual])
        incorrect = total - correct
        accuracy = correct / total if total > 0 else 0.0
        print(f"Classe '{actual}': réel={total}, correct={correct}, incorrect={incorrect}, accuracy={accuracy:.4f}")
        row = df_cm.loc[actual]
        row_str = "; ".join(f"{pred}={int(cnt)}" for pred, cnt in row.items() if cnt > 0)
        print(f"  Prédictions par classe : {row_str}")

    print("\n=== Matrice de confusion (Réel x Prédit) ===")
    print(df_cm.to_string())


def save_per_class_prediction_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
    output_dir: str = "outputs",
) -> None:
    """Enregistre les résultats d'évaluation par classe en fichiers CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    y_true = pd.Series(y_true, name="Actual")
    y_pred = pd.Series(y_pred, name="Predicted")
    labels = sorted(pd.unique(pd.concat([y_true, y_pred])))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    # Créer le résumé par classe
    summary_data = []
    for actual in labels:
        total = int(df_cm.loc[actual].sum())
        correct = int(df_cm.loc[actual, actual])
        incorrect = total - correct
        accuracy = correct / total if total > 0 else 0.0
        summary_data.append({
            "Classe": actual,
            "Réel": total,
            "Correct": correct,
            "Incorrect": incorrect,
            "Accuracy": round(accuracy, 4),
        })
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, f"per_class_summary_{model_name}.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\n✓ Résumé par classe enregistré : {summary_path}")

    # Enregistrer la matrice de confusion
    cm_path = os.path.join(output_dir, f"confusion_matrix_{model_name}.csv")
    df_cm.to_csv(cm_path)
    print(f"✓ Matrice de confusion enregistrée : {cm_path}")
