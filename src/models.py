"""
Module de définition et d'entraînement des modèles ML.

Modèles imposés par le sujet :
  - Random Forest
  - SVM (LinearSVC et SVC RBF)
  - MLP (réseau de neurones simple)
"""
import time
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC


def train_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    name: str = "Modèle",
) -> Tuple[Any, float]:
    """
    Entraîne un modèle en mesurant le temps.

    Parameters
    ----------
    model : Any
        Instance du modèle à entraîner.
    X_train : pd.DataFrame
        Features d'entraînement.
    y_train : pd.Series
        Label d'entraînement.
    name : str
        Nom du modèle (pour l'affichage).

    Returns
    -------
    model : Any
        Le modèle entraîné.
    training_time : float
        Le temps d'entraînement en secondes.
    """
    print(f"\n--- Entraînement {name} ---")
    t0 = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - t0
    print(f"  ✔ Terminé en {training_time:.2f} s")
    return model, training_time


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
) -> Tuple[RandomForestClassifier, float]:
    """Entraîne un Random Forest."""
    model = RandomForestClassifier(**params)
    return train_model(model, X_train, y_train, name="Random Forest")


def train_svm_linear(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
) -> Tuple[LinearSVC, float]:
    """Entraîne un SVM linéaire (LinearSVC)."""
    model = LinearSVC(**params)
    return train_model(model, X_train, y_train, name="SVM (LinearSVC)")


def train_svm_rbf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
    sample_size: int = 20000,
    random_state: int = 42,
) -> Tuple[SVC, float]:
    """
    Entraîne un SVM avec noyau RBF sur un sous-échantillon stratifié.

    Le SVM RBF a une complexité O(n²) à O(n³) qui le rend impraticable
    sur des datasets de grande taille. On entraîne donc sur un échantillon
    réduit, en stratifiant pour préserver la distribution des classes.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Données d'entraînement complètes.
    params : dict
        Hyperparamètres du SVC.
    sample_size : int
        Taille de l'échantillon.
    random_state : int
        Graine aléatoire.

    Returns
    -------
    model : SVC
    training_time : float
    """
    # Échantillon stratifié
    X_sample, _, y_sample, _ = train_test_split(
        X_train, y_train,
        train_size=sample_size,
        random_state=random_state,
        stratify=y_train,
    )

    print(f"\n  → SVM RBF entraîné sur {sample_size:,} échantillons (sur {X_train.shape[0]:,})")
    model = SVC(**params)
    return train_model(model, X_sample, y_sample, name="SVM (RBF)")


def train_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
) -> Tuple[MLPClassifier, float]:
    """Entraîne un MLP (perceptron multi-couches)."""
    model = MLPClassifier(**params)
    model, t = train_model(model, X_train, y_train, name="MLP (réseau de neurones)")
    print(f"  Epochs effectués : {model.n_iter_}")
    return model, t


def get_predictions(model: Any, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule les prédictions et les scores de probabilité (ou decision_function).

    Parameters
    ----------
    model : Any
        Modèle entraîné.
    X_test : pd.DataFrame
        Features de test.

    Returns
    -------
    y_pred : np.ndarray
        Prédictions binaires (0/1).
    y_score : np.ndarray
        Probabilités d'appartenance à la classe positive (ou decision_function
        pour LinearSVC qui n'a pas predict_proba).
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # Cas LinearSVC : on retourne la decision_function (utilisable pour ROC-AUC)
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred.astype(float)

    return y_pred, y_score
