"""
Module de prétraitement des données.

Couvre les étapes du point 2 du sujet :
  - Nettoyage et sélection des variables pertinentes
  - Séparation des données (entraînement / test)

Étapes :
  1. clean_data : suppression Inf, NaN, doublons
  2. prepare_features_labels : conversion en label binaire
  3. split_data : split stratifié train/test
  4. select_features : sélection en 3 étapes (variance, corrélation, importance)
  5. normalize_features : standardisation (fit sur train uniquement)
"""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Nettoie le dataset : remplace les Inf par NaN, supprime les lignes incomplètes
    et les doublons.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset brut.

    Returns
    -------
    pd.DataFrame
        Dataset nettoyé.
    dict
        Statistiques sur le nettoyage (nombre de lignes supprimées à chaque étape).
    """
    n_initial = df.shape[0]

    # 1. Inf -> NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # 2. Suppression des NaN
    df = df.dropna()
    n_after_nan = df.shape[0]

    # 3. Suppression des doublons
    df = df.drop_duplicates()
    n_after_dup = df.shape[0]

    # Reset de l'index
    df = df.reset_index(drop=True)

    stats = {
        "n_initial": n_initial,
        "n_dropped_nan": n_initial - n_after_nan,
        "n_dropped_dup": n_after_nan - n_after_dup,
        "n_final": n_after_dup,
        "retention_rate": n_after_dup / n_initial * 100,
    }

    print(f"\n=== Nettoyage ===")
    print(f"  Initial            : {stats['n_initial']:,}")
    print(f"  - NaN supprimés    : {stats['n_dropped_nan']:,}")
    print(f"  - Doublons         : {stats['n_dropped_dup']:,}")
    print(f"  Final              : {stats['n_final']:,} ({stats['retention_rate']:.2f}%)")

    return df, stats


def prepare_features_labels(
    df: pd.DataFrame,
    label_column: str = "Label",
    benign_label: str = "BENIGN",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Sépare les features (X) du label (y) et convertit le label en binaire.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset nettoyé.
    label_column : str
        Nom de la colonne contenant le label.
    benign_label : str
        Valeur du label correspondant au trafic normal.

    Returns
    -------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Label binaire (0 = BENIGN, 1 = attaque).
    """
    y = (df[label_column] != benign_label).astype(int)
    X = df.drop(columns=[label_column])

    print(f"\n=== Préparation features/label ===")
    print(f"  Features : {X.shape}")
    print(f"  Label    : {y.shape}")
    print(f"  Distribution : {dict(y.value_counts())}")

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split stratifié train/test.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Label.
    test_size : float
        Proportion du test set.
    random_state : int
        Graine aléatoire.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    print(f"\n=== Split stratifié ({int((1-test_size)*100)}/{int(test_size*100)}) ===")
    print(f"  Train : {X_train.shape[0]:,} — attaques : {y_train.mean()*100:.2f}%")
    print(f"  Test  : {X_test.shape[0]:,} — attaques : {y_test.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test


def select_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    variance_threshold: float = 0.0,
    correlation_threshold: float = 0.95,
    rf_selector_params: dict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Sélection des features en 3 étapes :
      1. Suppression des features à variance nulle
      2. Suppression des features fortement corrélées
      3. Calcul de l'importance par Random Forest (informatif)

    Les sélections sont calculées sur le train uniquement, puis appliquées au test.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Features train et test.
    y_train : pd.Series
        Label train (pour le calcul d'importance).
    variance_threshold : float
        Seuil de variance.
    correlation_threshold : float
        Seuil de corrélation.
    rf_selector_params : dict
        Paramètres du Random Forest utilisé pour calculer l'importance.

    Returns
    -------
    X_train_selected, X_test_selected : pd.DataFrame
        Features filtrées.
    feature_importances : pd.DataFrame
        Tableau des importances triées.
    stats : dict
        Statistiques sur la sélection (nombre de features à chaque étape).
    """
    if rf_selector_params is None:
        rf_selector_params = {"n_estimators": 50, "max_depth": 15, "n_jobs": -1}

    n_initial = X_train.shape[1]

    # 1. Variance Threshold
    var_selector = VarianceThreshold(threshold=variance_threshold)
    var_selector.fit(X_train)
    kept_var = X_train.columns[var_selector.get_support()].tolist()
    dropped_var = X_train.columns[~var_selector.get_support()].tolist()
    X_train_v1 = X_train[kept_var]
    X_test_v1 = X_test[kept_var]

    # 2. Corrélation
    corr_matrix = X_train_v1.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    dropped_corr = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
    X_train_v2 = X_train_v1.drop(columns=dropped_corr)
    X_test_v2 = X_test_v1.drop(columns=dropped_corr)

    # 3. Importance Random Forest (informatif uniquement, on garde toutes les features)
    rf_sel = RandomForestClassifier(**rf_selector_params)
    rf_sel.fit(X_train_v2, y_train)

    feature_importances = pd.DataFrame({
        "Feature": X_train_v2.columns,
        "Importance": rf_sel.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    feature_importances["Cumulative"] = feature_importances["Importance"].cumsum()
    n_95 = (feature_importances["Cumulative"] <= 0.95).sum() + 1

    stats = {
        "n_initial": n_initial,
        "n_dropped_variance": len(dropped_var),
        "n_dropped_correlation": len(dropped_corr),
        "n_final": X_train_v2.shape[1],
        "n_features_95pct": int(n_95),
        "dropped_variance": dropped_var,
        "dropped_correlation": dropped_corr,
    }

    print(f"\n=== Sélection de features ===")
    print(f"  Features initiales        : {stats['n_initial']}")
    print(f"  - Variance nulle          : -{stats['n_dropped_variance']}")
    print(f"  - Corrélation > {correlation_threshold} : -{stats['n_dropped_correlation']}")
    print(f"  Features finales          : {stats['n_final']}")
    print(f"  → {stats['n_features_95pct']} features couvrent 95% de l'importance.")

    return X_train_v2, X_test_v2, feature_importances, stats


def normalize_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Normalisation StandardScaler. Le scaler est ajusté sur le train uniquement
    pour éviter toute fuite d'information.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Features train et test.

    Returns
    -------
    X_train_scaled, X_test_scaled : pd.DataFrame
        Features normalisées (DataFrame pour conserver les noms de colonnes).
    scaler : StandardScaler
        Le transformateur ajusté (pour la reproductibilité ou un usage ultérieur).
    """
    scaler = StandardScaler()
    X_train_arr = scaler.fit_transform(X_train)
    X_test_arr = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_arr, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_arr, columns=X_test.columns, index=X_test.index)

    print(f"\n=== Normalisation ===")
    print(f"  Moyenne moyenne (train)    : {X_train_scaled.mean().mean():.6f}")
    print(f"  Écart-type moyen (train)   : {X_train_scaled.std().mean():.6f}")

    return X_train_scaled, X_test_scaled, scaler
