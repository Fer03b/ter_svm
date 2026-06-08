"""
Pipeline complet du projet TER — Classification MULTI-CLASSE avec CICIDS2017.

Lance toutes les étapes dans l'ordre pour une classification multi-classe :
  1. Chargement de tous les fichiers CSV du dossier data/ (par portions)
  2. Exploration des données
  3. Nettoyage
  4. Préparation features/label MULTI-CLASSE + split stratifié + sélection de features
  5. Normalisation
  6. Entraînement des modèles (Random Forest, SVM, MLP)
  7. Évaluation et comparaison
  8. Sauvegarde des résultats

Usage :
    python main_multiclass.py

Note : Les labels sont conservés en texte (ex: "BENIGN", "DDoS", "Botnet")
       au lieu d'être convertis en binaire (0, 1).
"""
from pathlib import Path
import joblib

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
    train_mlp,
    train_random_forest,
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


def main():
    """Pipeline complet multi-classe."""

    # ============================================================
    # 1. CHARGEMENT ET EXPLORATION (MULTI-CLASSE)
    # ============================================================
    print("\n" + "=" * 70)
    print("  ÉTAPE 1 — Chargement MULTI-CLASSE et exploration des données")
    print("=" * 70)
    print("Note : Chargement de TOUS les fichiers .csv du dossier data/")
    print("       par portions de 50 000 lignes pour limiter la RAM.\n")

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

    # Random Forest
    print("Entraînement du Random Forest...")
    model_rf, _ = train_random_forest(
        X_train_norm, y_train,
        {
            "n_estimators": 100,
            "max_depth": 20,
            "n_jobs": -1,
            "random_state": config.RANDOM_STATE,
        }
    )
    joblib.dump(model_rf, config.MODELS_DIR / "random_forest_multiclass.joblib")

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

    # MLP
    print("Entraînement du MLP...")
    model_mlp, _ = train_mlp(
        X_train_norm, y_train,
        {
            "hidden_layer_sizes": (128, 64),
            "max_iter": 500,
            "solver": "adam",
            "random_state": config.RANDOM_STATE,
        }
    )
    joblib.dump(model_mlp, config.MODELS_DIR / "mlp_multiclass.joblib")

    # ============================================================
    # 5. ÉVALUATION ET COMPARAISON
    # ============================================================
    print("\n" + "=" * 70)
    print("  ÉTAPE 7 — Évaluation et comparaison des modèles")
    print("=" * 70)

    # Prédictions
    y_pred_rf, _ = get_predictions(model_rf, X_test_norm)
    y_pred_svm_lin, _ = get_predictions(model_svm_lin, X_test_norm)
    y_pred_svm_rbf, _ = get_predictions(model_svm_rbf, X_test_norm)
    y_pred_mlp, _ = get_predictions(model_mlp, X_test_norm)

    # Évaluation de chaque modèle (avec average="weighted" pour multi-classe)
    print("\n" + "-" * 70)
    print("Random Forest")
    print("-" * 70)
    metrics_rf = evaluate_model(y_test, y_pred_rf, average="weighted")
    print_classification_report(y_test, y_pred_rf)

    print("\n" + "-" * 70)
    print("SVM Linéaire")
    print("-" * 70)
    metrics_svm_lin = evaluate_model(y_test, y_pred_svm_lin, average="weighted")
    print_classification_report(y_test, y_pred_svm_lin)

    print("\n" + "-" * 70)
    print("SVM RBF")
    print("-" * 70)
    metrics_svm_rbf = evaluate_model(y_test, y_pred_svm_rbf, average="weighted")
    print_classification_report(y_test, y_pred_svm_rbf)

    print("\n" + "-" * 70)
    print("MLP")
    print("-" * 70)
    metrics_mlp = evaluate_model(y_test, y_pred_mlp, average="weighted")
    print_classification_report(y_test, y_pred_mlp)

    # Tableau de comparaison
    print("\n" + "=" * 70)
    print("  COMPARAISON DES MODÈLES")
    print("=" * 70)

    comparison_df = build_comparison_table(
        {
            "Random Forest": metrics_rf,
            "SVM Linear": metrics_svm_lin,
            "SVM RBF": metrics_svm_rbf,
            "MLP": metrics_mlp,
        }
    )
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
            "Random Forest": y_pred_rf,
            "SVM Linear": y_pred_svm_lin,
            "SVM RBF": y_pred_svm_rbf,
            "MLP": y_pred_mlp,
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

    # Importance des features (Random Forest)
    print("Génération du graphique d'importance des features...")
    plot_feature_importance(
        feature_importances,
        top_n=20,
        output_path=config.FIGURES_DIR / "feature_importance_multiclass.png",
    )

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
        "Accuracy": metrics_rf["Accuracy"],
        "Precision (weighted)": metrics_rf["Precision"],
        "Recall (weighted)": metrics_rf["Recall"],
        "F1-score (weighted)": metrics_rf["F1-score"],
    }

    print("\n📊 Résumé du pipeline :")
    for key, value in summary.items():
        print(f"  {key:<25} : {value}")

    # Sauvegarde du scaler
    joblib.dump(scaler, config.MODELS_DIR / "scaler_multiclass.joblib")
    print(f"\n✅ Modèles sauvegardés dans {config.MODELS_DIR}/")
    print(f"✅ Résultats sauvegardés dans {config.RESULTS_DIR}/")
    print(f"✅ Figures sauvegardées dans {config.FIGURES_DIR}/")

    print("\n" + "=" * 70)
    print("  ✅ PIPELINE MULTI-CLASSE COMPLÉTÉ AVEC SUCCÈS !")
    print("=" * 70)


if __name__ == "__main__":
    main()
