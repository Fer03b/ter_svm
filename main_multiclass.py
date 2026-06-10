"""
Pipeline complet du projet TER — Classification MULTI-CLASSE avec CICIDS2017.

Lance toutes les étapes dans l'ordre pour une classification multi-classe :
  1. Chargement de tous les fichiers CSV du dossier data/ (par portions)
  2. Exploration des données
  3. Nettoyage
  4. Préparation features/label MULTI-CLASSE + split stratifié + sélection de features
  5. Normalisation
  6. Entraînement des modèles (SVM linéaire et RBF)
  7. Évaluation et comparaison
  8. Sauvegarde des résultats

Usage :
    python main_multiclass.py

Note : Les labels sont conservés en texte (ex: "BENIGN", "DDoS", "Botnet")
       au lieu d'être convertis en binaire (0, 1).
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

import config
from src.data_loader import (
    describe_dataset,
    load_dataset_multi_class,  # ✨ NOUVEAU : charge tous les fichiers .csv
    print_dataset_summary,
)
from src.evaluation import (
    build_comparison_table,
    evaluate_model,
    identify_best_model,
    print_classification_report,
    print_comparison_table,
)
from src.models import (
    get_predictions,
    train_svm_linear,
    train_svm_rbf,
)
from src.preprocessing import (
    clean_data,
    normalize_features,
    prepare_features_labels_multiclass,  # ✨ NOUVEAU : labels textuels
    select_features,
    split_data_multiclass,  # ✨ NOUVEAU : split multi-classe stratifié
)
from src.visualization import (
    plot_class_distribution,
    plot_confusion_matrices,
    plot_correlation_matrix,
    plot_feature_importance,
    plot_metrics_comparison,
    plot_roc_curves,
)
from src.extra_visualization import plot_prediction_distribution


def print_metrics_table(metrics: dict, model_name: str) -> None:
    """Affiche un tableau lisible pour weighted / macro metrics."""
    df = pd.DataFrame(metrics).T.round(4)
    print(f"\n=== {model_name} : métriques détaillées ===")
    print(df.to_string())


def load_prepared_dataset(
    x_train_path: Path,
    x_test_path: Path,
    y_train_path: Path,
    y_test_path: Path,
):
    """Charge les jeux X_train, X_test, y_train, y_test déjà préparés."""
    def _read(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        if path.suffix == ".csv":
            df = pd.read_csv(path)
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
            return df
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix in (".pkl", ".joblib"):
            return joblib.load(path)
        raise ValueError(f"Format de fichier non supporté : {path.suffix}")

    X_train = _read(x_train_path)
    X_test = _read(x_test_path)
    y_train = _read(y_train_path)
    y_test = _read(y_test_path)

    if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]
    if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]

    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)
    if isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test)

    return X_train, X_test, y_train, y_test


def main():
    """Pipeline complet multi-classe."""

    # ============================================================
    # 1. CHARGEMENT ET EXPLORATION (MULTI-CLASSE)
    # ============================================================
    print("\n" + "=" * 70)
    print("  ÉTAPE 1 — Chargement MULTI-CLASSE et exploration des données")
    print("=" * 70)
    prepared_mode = False

    if config.USE_PREPARED_DATASET:
        print("Note : Utilisation d'un dataset déjà nettoyé, divisé et normalisé.")
        X_train_norm, X_test_norm, y_train, y_test = load_prepared_dataset(
            config.PREPARED_X_TRAIN,
            config.PREPARED_X_TEST,
            config.PREPARED_Y_TRAIN,
            config.PREPARED_Y_TEST,
        )
        print(f"  X_train : {X_train_norm.shape}")
        print(f"  X_test  : {X_test_norm.shape}")
        print(f"  y_train : {y_train.shape}")
        print(f"  y_test  : {y_test.shape}")
        print(f"  Classes : {sorted(y_train.unique())}")
        feature_importances = None
        scaler = None
        prepared_mode = True
    else:
        # ✨ NOUVEAU : Charge tous les fichiers .csv du dossier data/ par portions
        df = load_dataset_multi_class(
            data_dir=config.DATA_DIR,
            label_column=config.LABEL_COLUMN,
            chunk_size=50000,  # 50 000 lignes par fichier
            random_state=config.RANDOM_STATE,
        )

        stats_initial = describe_dataset(df, label_column=config.LABEL_COLUMN)
        print_dataset_summary(stats_initial)

        # Figure : distribution des classes
        print("\nGénération des figures EDA...")
        plot_class_distribution(
            df,
            label_column=config.LABEL_COLUMN,
            output_path=config.FIGURES_DIR / "01_class_distribution_multiclass.png",
            dpi=150,
        )

        # ============================================================
        # 2. NETTOYAGE
        # ============================================================
        print("\n" + "=" * 70)
        print("  ÉTAPE 2 — Nettoyage des données")
        print("=" * 70)

        df, stats_clean = clean_data(df)

        # ============================================================
        # 3. PRÉPARATION + SPLIT + SÉLECTION DE FEATURES + NORMALISATION
        # ============================================================
        print("\n" + "=" * 70)
        print("  ÉTAPE 3 — Préparation MULTI-CLASSE et split stratifié")
        print("=" * 70)

        # ✨ NOUVEAU : Prépare features/label en GARDANT les labels textuels
        X, y = prepare_features_labels_multiclass(
            df,
            label_column=config.LABEL_COLUMN,
        )

        # ✨ NOUVEAU : Split stratifié qui fonctionne avec labels textuels
        X_train, X_test, y_train, y_test = split_data_multiclass(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
        )

        # Sélection de features (identique au pipeline binaire)
        print("\n" + "=" * 70)
        print("  ÉTAPE 4 — Sélection de features")
        print("=" * 70)

        X_train_sel, X_test_sel, feature_importances, stats_feat = select_features(
            X_train, X_test, y_train,
            variance_threshold=0.0,
            correlation_threshold=config.CORRELATION_THRESHOLD,
        )

        # Normalisation (identique au pipeline binaire)
        print("\n" + "=" * 70)
        print("  ÉTAPE 5 — Normalisation des features")
        print("=" * 70)

        X_train_norm, X_test_norm, scaler = normalize_features(X_train_sel, X_test_sel)

    # ============================================================
    # 4. ENTRAÎNEMENT DES MODÈLES (MULTI-CLASSE)
    # ============================================================
    print("\n" + "=" * 70)
    print("  ÉTAPE 6 — Entraînement des modèles (MULTI-CLASSE)")
    print("=" * 70)
    print("Note : Les modèles gèrent nativement la classification multi-classe.\n")

    # Seuls les SVM sont activés pour ce test.
    # Random Forest et MLP sont désactivés/commentés pour accélérer l'expérimentation.

    # SVM Linéaire
    print("Entraînement du SVM (kernel linéaire)...")
    model_svm_lin, _ = train_svm_linear(
        X_train_norm, y_train,
        {
            "C": 1.0,
            "random_state": config.RANDOM_STATE,
            "max_iter": 1000,
        }
    )
    joblib.dump(model_svm_lin, config.MODELS_DIR / "svm_linear_multiclass.joblib")

    # SVM RBF
    print("Entraînement du SVM (kernel RBF)...")
    model_svm_rbf, _ = train_svm_rbf(
        X_train_norm, y_train,
        {
            "C": 1.0,
            "gamma": "scale",
            "probability": True,
            "random_state": config.RANDOM_STATE,
        }
    )
    joblib.dump(model_svm_rbf, config.MODELS_DIR / "svm_rbf_multiclass.joblib")

    # ============================================================
    # 5. ÉVALUATION ET COMPARAISON
    # ============================================================
    print("\n" + "=" * 70)
    print("  ÉTAPE 7 — Évaluation et comparaison des modèles")
    print("=" * 70)

    # Prédictions
    y_pred_svm_lin, _ = get_predictions(model_svm_lin, X_test_norm)
    y_pred_svm_rbf, _ = get_predictions(model_svm_rbf, X_test_norm)

    # Distribution des prédictions et précision par classe (utile pour comparer datasets)
    try:
        plot_prediction_distribution(
            {
                "SVM Linear": y_pred_svm_lin,
                "SVM RBF": y_pred_svm_rbf,
            },
            y_test,
            output_path=config.FIGURES_DIR / "prediction_distribution_multiclass.png",
        )
    except Exception as e:
        print(f"  ⚠ Impossible de générer prediction_distribution: {e}")

    # Évaluation de chaque modèle (avec weighted et macro pour multi-classe)
    print("\n" + "-" * 70)
    print("SVM Linéaire")
    print("-" * 70)
    metrics_svm_lin = evaluate_model(
        y_test, y_pred_svm_lin, average=("weighted", "macro")
    )
    print_classification_report(y_test, y_pred_svm_lin)
    print_metrics_table(metrics_svm_lin, "SVM Linéaire")

    print("\n" + "-" * 70)
    print("SVM RBF")
    print("-" * 70)
    metrics_svm_rbf = evaluate_model(
        y_test, y_pred_svm_rbf, average=("weighted", "macro")
    )
    print_classification_report(y_test, y_pred_svm_rbf)
    print_metrics_table(metrics_svm_rbf, "SVM RBF")

    # Tableau de comparaison
    print("\n" + "=" * 70)
    print("  COMPARAISON DES MODÈLES")
    print("=" * 70)

    comparison_df = build_comparison_table(
        {
            "SVM Linear": metrics_svm_lin["weighted"],
            "SVM RBF": metrics_svm_rbf["weighted"],
        }
    )

    print("\n=== Résumé comparatif des métriques weighted / macro ===")
    combined_metrics = {
        "SVM Linear (weighted)": metrics_svm_lin["weighted"],
        "SVM Linear (macro)": metrics_svm_lin["macro"],
        "SVM RBF (weighted)": metrics_svm_rbf["weighted"],
        "SVM RBF (macro)": metrics_svm_rbf["macro"],
    }
    print(pd.DataFrame(combined_metrics).T.round(4).to_string())
    print_comparison_table(comparison_df)
    comparison_df.to_csv(
        config.RESULTS_DIR / "comparison_table_multiclass.csv",
        index=False,
    )

    # Identification du meilleur modèle
    best_model_name = identify_best_model(comparison_df, metric="F1-score")
    print(f"\n✅ Meilleur modèle : {best_model_name}")

    # ============================================================
    # 6. GÉNÉRATIONS DES FIGURES
    # ============================================================
    print("\n" + "=" * 70)
    print("  ÉTAPE 8 — Génération des figures")
    print("=" * 70)

    # Matrices de confusion
    print("Génération des matrices de confusion...")
    plot_confusion_matrices(
        {
            "SVM Linear": y_pred_svm_lin,
            "SVM RBF": y_pred_svm_rbf,
        },
        y_test,
        output_path=config.FIGURES_DIR / "confusion_matrices_multiclass.png",
    )

    # Comparaison des métriques
    print("Génération du graphique de comparaison des métriques...")
    plot_metrics_comparison(
        comparison_df,
        output_path=config.FIGURES_DIR / "metrics_comparison_multiclass.png",
    )

    if not prepared_mode:
        # Importance des features (Random Forest)
        print("Génération du graphique d'importance des features...")
        plot_feature_importance(
            feature_importances,
            top_n=20,
            output_path=config.FIGURES_DIR / "feature_importance_multiclass.png",
        )
    else:
        print("Génération du graphique d'importance des features : ignoré en mode dataset préparé.")

    # Courbes ROC (pour multi-classe : one-vs-rest)
    # Note : Les courbes ROC multi-classe nécessitent une approche spécifique (one-vs-rest)
    # Cette fonctionnalité sera développée dans une version ultérieure
    # print("Génération des courbes ROC (one-vs-rest)...")
    # plot_roc_curves(
    #     y_test, y_pred_rf,
    #     model_rf, X_test_norm,
    #     output_path=config.FIGURES_DIR / "roc_curves_multiclass.png",
    # )

    # ============================================================
    # 7. SAUVEGARDE DES RÉSULTATS
    # ============================================================
    print("\n" + "=" * 70)
    print("  ÉTAPE 9 — Sauvegarde des résultats")
    print("=" * 70)

    # Résumé du pipeline
    best_metrics = metrics_svm_lin if best_model_name == "SVM Linear" else metrics_svm_rbf
    if prepared_mode:
        summary = {
            "Pipeline": "Multi-classe (dataset préparé)",
            "Nb. samples": len(X_train_norm) + len(X_test_norm),
            "Nb. features (final)": X_train_norm.shape[1],
            "Nb. classes": y_train.nunique(),
            "Classes": sorted(y_train.unique()),
            "Train/Test split": f"{len(X_train_norm)}/{len(X_test_norm)}",
            "Meilleur modèle": best_model_name,
            "Accuracy": best_metrics["Accuracy"],
            "Precision (weighted)": best_metrics["Precision"],
            "Recall (weighted)": best_metrics["Recall"],
            "F1-score (weighted)": best_metrics["F1-score"],
        }
    else:
        summary = {
            "Pipeline": "Multi-classe",
            "Nb. fichiers chargés": len(list(config.DATA_DIR.glob("*.csv"))),
            "Nb. samples": len(df),
            "Nb. features (initial)": X.shape[1],
            "Nb. features (final)": X_train_norm.shape[1],
            "Nb. classes": y.nunique(),
            "Classes": sorted(y.unique()),
            "Train/Test split": f"{len(X_train)}/{len(X_test)}",
            "Meilleur modèle": best_model_name,
            "Accuracy": best_metrics["Accuracy"],
            "Precision (weighted)": best_metrics["Precision"],
            "Recall (weighted)": best_metrics["Recall"],
            "F1-score (weighted)": best_metrics["F1-score"],
        }

    print("\n📊 Résumé du pipeline :")
    for key, value in summary.items():
        print(f"  {key:<25} : {value}")

    if scaler is not None:
        joblib.dump(scaler, config.MODELS_DIR / "scaler_multiclass.joblib")
    else:
        print("\n✅ Scaler non sauvegardé : dataset déjà préparé externe.")
    print(f"\n✅ Modèles sauvegardés dans {config.MODELS_DIR}/")
    print(f"✅ Résultats sauvegardés dans {config.RESULTS_DIR}/")
    print(f"✅ Figures sauvegardées dans {config.FIGURES_DIR}/")

    print("\n" + "=" * 70)
    print("  ✅ PIPELINE MULTI-CLASSE COMPLÉTÉ AVEC SUCCÈS !")
    print("=" * 70)


if __name__ == "__main__":
    main()
