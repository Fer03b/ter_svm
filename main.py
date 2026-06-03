"""
Pipeline complet du projet TER — Détection d'attaques réseau par ML.

Lance toutes les étapes dans l'ordre :
  1. Chargement et exploration des données
  2. Nettoyage
  3. Préparation features/label + split + sélection de features + normalisation
  4. Entraînement des 4 modèles (RF, SVM linéaire, SVM RBF, MLP)
  5. Évaluation et comparaison
  6. Génération des figures
  7. Sauvegarde des résultats

Usage :
    python main.py
"""
import joblib

import config
from src.data_loader import (
    describe_dataset,
    load_dataset,
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
    prepare_features_labels,
    select_features,
    split_data,
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
    """Pipeline complet."""

    # ============================================================
    # 1. CHARGEMENT ET EXPLORATION
    # ============================================================
    print("\n" + "=" * 60)
    print("  ÉTAPE 1 — Chargement et exploration des données")
    print("=" * 60)

    df = load_dataset(config.DATA_PATH)
    stats_initial = describe_dataset(df, label_column=config.LABEL_COLUMN)
    print_dataset_summary(stats_initial)

    # Figure : distribution des classes
    print("\nGénération des figures EDA...")
    plot_class_distribution(
        df,
        label_column=config.LABEL_COLUMN,
        output_path=config.FIGURES_DIR / "01_class_distribution.png",
        dpi=config.FIGURE_DPI,
    )

    # ============================================================
    # 2. NETTOYAGE
    # ============================================================
    print("\n" + "=" * 60)
    print("  ÉTAPE 2 — Nettoyage des données")
    print("=" * 60)

    df, stats_clean = clean_data(df)

    # ============================================================
    # 3. PRÉPARATION + SPLIT + SÉLECTION DE FEATURES + NORMALISATION
    # ============================================================
    print("\n" + "=" * 60)
    print("  ÉTAPE 3 — Préparation, split et sélection de features")
    print("=" * 60)

    X, y = prepare_features_labels(
        df,
        label_column=config.LABEL_COLUMN,
        benign_label=config.BENIGN_LABEL,
    )

    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    X_train, X_test, feature_importances, stats_select = select_features(
        X_train, X_test, y_train,
        variance_threshold=config.VARIANCE_THRESHOLD,
        correlation_threshold=config.CORRELATION_THRESHOLD,
        rf_selector_params=config.RF_SELECTOR_PARAMS,
    )

    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)

    # Figures additionnelles
    print("\nGénération des figures (corrélation et importance des features)...")
    plot_correlation_matrix(
        X_train,
        output_path=config.FIGURES_DIR / "02_correlation_matrix.png",
        n_features=30,
        dpi=config.FIGURE_DPI,
    )
    plot_feature_importance(
        feature_importances,
        output_path=config.FIGURES_DIR / "03_feature_importance.png",
        top_n=20,
        dpi=config.FIGURE_DPI,
    )

    # Sauvegarde de l'importance des features
    feature_importances.to_csv(
        config.RESULTS_DIR / "feature_importances.csv",
        index=False,
    )

    # ============================================================
    # 4. ENTRAÎNEMENT DES MODÈLES
    # ============================================================
    print("\n" + "=" * 60)
    print("  ÉTAPE 4 — Entraînement des modèles")
    print("=" * 60)

    # Random Forest
    rf, t_rf = train_random_forest(X_train_scaled, y_train, config.RF_PARAMS)
    y_pred_rf, y_score_rf = get_predictions(rf, X_test_scaled)
    print_classification_report("Random Forest", y_test, y_pred_rf, y_score_rf)

    # SVM linéaire
    svm_lin, t_svm_lin = train_svm_linear(X_train_scaled, y_train, config.SVM_LINEAR_PARAMS)
    y_pred_svm_lin, y_score_svm_lin = get_predictions(svm_lin, X_test_scaled)
    print_classification_report("SVM (LinearSVC)", y_test, y_pred_svm_lin, y_score_svm_lin)

    # SVM RBF (sur échantillon)
    svm_rbf, t_svm_rbf = train_svm_rbf(
        X_train_scaled, y_train,
        params=config.SVM_RBF_PARAMS,
        sample_size=config.SVM_RBF_SAMPLE_SIZE,
        random_state=config.RANDOM_STATE,
    )
    y_pred_svm_rbf, y_score_svm_rbf = get_predictions(svm_rbf, X_test_scaled)
    print_classification_report("SVM (RBF)", y_test, y_pred_svm_rbf, y_score_svm_rbf)

    # MLP
    mlp, t_mlp = train_mlp(X_train_scaled, y_train, config.MLP_PARAMS)
    y_pred_mlp, y_score_mlp = get_predictions(mlp, X_test_scaled)
    print_classification_report("MLP", y_test, y_pred_mlp, y_score_mlp)

    # ============================================================
    # 5. ÉVALUATION ET COMPARAISON
    # ============================================================
    print("\n" + "=" * 60)
    print("  ÉTAPE 5 — Évaluation comparative")
    print("=" * 60)

    results = [
        evaluate_model("Random Forest", y_test, y_pred_rf, y_score_rf, t_rf),
        evaluate_model("SVM (LinearSVC)", y_test, y_pred_svm_lin, y_score_svm_lin, t_svm_lin),
        evaluate_model("SVM (RBF, 20k)", y_test, y_pred_svm_rbf, y_score_svm_rbf, t_svm_rbf),
        evaluate_model("MLP", y_test, y_pred_mlp, y_score_mlp, t_mlp),
    ]

    comparison_df = build_comparison_table(results)
    print_comparison_table(comparison_df)
    identify_best_model(comparison_df, metric="F1-score")

    # Sauvegarde du tableau
    comparison_df.to_csv(config.RESULTS_DIR / "comparison_table.csv", index=False)

    # ============================================================
    # 6. FIGURES COMPARATIVES
    # ============================================================
    print("\n" + "=" * 60)
    print("  ÉTAPE 6 — Génération des figures comparatives")
    print("=" * 60)

    roc_data = [
        ("Random Forest", y_score_rf),
        ("SVM (LinearSVC)", y_score_svm_lin),
        ("SVM (RBF)", y_score_svm_rbf),
        ("MLP", y_score_mlp),
    ]
    plot_roc_curves(
        roc_data, y_test,
        output_path=config.FIGURES_DIR / "04_roc_curves.png",
        dpi=config.FIGURE_DPI,
    )

    predictions = {
        "Random Forest": y_pred_rf,
        "SVM (LinearSVC)": y_pred_svm_lin,
        "SVM (RBF)": y_pred_svm_rbf,
        "MLP": y_pred_mlp,
    }
    plot_confusion_matrices(
        predictions, y_test,
        output_path=config.FIGURES_DIR / "05_confusion_matrices.png",
        dpi=config.FIGURE_DPI,
    )

    plot_metrics_comparison(
        comparison_df,
        output_path=config.FIGURES_DIR / "06_metrics_comparison.png",
        dpi=config.FIGURE_DPI,
    )

    # ============================================================
    # 7. SAUVEGARDE DES MODÈLES
    # ============================================================
    print("\n" + "=" * 60)
    print("  ÉTAPE 7 — Sauvegarde des modèles")
    print("=" * 60)

    joblib.dump(rf, config.MODELS_DIR / "random_forest.joblib")
    joblib.dump(svm_lin, config.MODELS_DIR / "svm_linear.joblib")
    joblib.dump(svm_rbf, config.MODELS_DIR / "svm_rbf.joblib")
    joblib.dump(mlp, config.MODELS_DIR / "mlp.joblib")
    joblib.dump(scaler, config.MODELS_DIR / "scaler.joblib")
    print(f"  ✔ 4 modèles + scaler sauvegardés dans {config.MODELS_DIR}/")

    # ============================================================
    # RÉCAPITULATIF FINAL
    # ============================================================
    print("\n" + "=" * 60)
    print("  ✅ PIPELINE TERMINÉ AVEC SUCCÈS")
    print("=" * 60)
    print(f"\n  Résultats disponibles :")
    print(f"    - Figures : {config.FIGURES_DIR}")
    print(f"    - Modèles : {config.MODELS_DIR}")
    print(f"    - Tableaux : {config.RESULTS_DIR}")
    print()


if __name__ == "__main__":
    main()
