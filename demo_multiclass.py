"""
Démonstration rapide du pipeline MULTI-CLASSE.

Ce script teste rapidement les nouvelles fonctionnalités sans nécessiter
un entraînement complet des modèles.

Usage :
    python demo_multiclass.py
"""
from pathlib import Path

import config
from src.data_loader import (
    describe_dataset,
    load_dataset_multi_class,
    print_dataset_summary,
)
from src.preprocessing import (
    clean_data,
    prepare_features_labels_multiclass,
    split_data_multiclass,
)


def demo():
    """Démonstration des nouvelles fonctionnalités."""

    print("\n" + "=" * 70)
    print("  DÉMONSTRATION : Pipeline MULTI-CLASSE")
    print("=" * 70)

    # ============================================================
    # 1. Chargement MULTI-CLASSE
    # ============================================================
    print("\n" + "-" * 70)
    print("1️⃣  Chargement MULTI-CLASSE (tous les fichiers .csv par portions)")
    print("-" * 70)

    print(f"📁 Dossier data/ : {config.DATA_DIR}")
    csv_files = sorted(config.DATA_DIR.glob("*.csv"))
    print(f"📄 Fichiers trouvés : {len(csv_files)}")
    for f in csv_files:
        print(f"   - {f.name}")

    print("\n⏳ Chargement par portions (50 000 lignes max par fichier)...")
    df = load_dataset_multi_class(
        data_dir=config.DATA_DIR,
        label_column=config.LABEL_COLUMN,
        chunk_size=50000,
        random_state=config.RANDOM_STATE,
    )

    print(f"\n✅ Chargement terminé !")
    print(f"   Shape : {df.shape}")
    print(f"   Mémoire utilisée : {df.memory_usage(deep=True).sum() / 1024**2:.2f} Mo")

    # ============================================================
    # 2. Exploration des données
    # ============================================================
    print("\n" + "-" * 70)
    print("2️⃣  Exploration des données")
    print("-" * 70)

    stats = describe_dataset(df, config.LABEL_COLUMN)
    print_dataset_summary(stats)

    # ============================================================
    # 3. Nettoyage
    # ============================================================
    print("\n" + "-" * 70)
    print("3️⃣  Nettoyage des données")
    print("-" * 70)

    df_clean, stats_clean = clean_data(df)

    # ============================================================
    # 4. Préparation MULTI-CLASSE
    # ============================================================
    print("\n" + "-" * 70)
    print("4️⃣  Préparation MULTI-CLASSE (labels textuels)")
    print("-" * 70)

    X, y = prepare_features_labels_multiclass(
        df_clean,
        label_column=config.LABEL_COLUMN,
    )

    print(f"\n📊 Informations sur les labels :")
    print(f"   Type des labels : {y.dtype}")
    print(f"   Échantillon de labels : {y.head(10).tolist()}")

    # ============================================================
    # 5. Split MULTI-CLASSE
    # ============================================================
    print("\n" + "-" * 70)
    print("5️⃣  Split stratifié MULTI-CLASSE")
    print("-" * 70)

    X_train, X_test, y_train, y_test = split_data_multiclass(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    # ============================================================
    # 6. Vérification de l'équilibre
    # ============================================================
    print("\n" + "-" * 70)
    print("6️⃣  Vérification de l'équilibre des classes")
    print("-" * 70)

    print("\n📊 Proportion des classes (TRAIN) :")
    train_props = y_train.value_counts(normalize=True) * 100
    for label in sorted(train_props.index):
        pct = train_props[label]
        print(f"   {label:<20} : {pct:>6.2f}%")

    print("\n📊 Proportion des classes (TEST) :")
    test_props = y_test.value_counts(normalize=True) * 100
    for label in sorted(test_props.index):
        pct = test_props[label]
        print(f"   {label:<20} : {pct:>6.2f}%")

    # ============================================================
    # 7. Résumé
    # ============================================================
    print("\n" + "=" * 70)
    print("  ✅ DÉMONSTRATION TERMINÉE")
    print("=" * 70)

    print("\n📋 Résumé :")
    print(f"   ✓ {len(csv_files)} fichier(s) CSV chargé(s)")
    print(f"   ✓ {len(df_clean):,} samples après nettoyage")
    print(f"   ✓ {X.shape[1]} features")
    print(f"   ✓ {y.nunique()} classes (multi-classe)")
    print(f"   ✓ Labels : {sorted(y.unique())}")
    print(f"   ✓ Split : {len(X_train):,} train / {len(X_test):,} test")
    print(f"   ✓ Équilibre stratifié ✅")

    print("\n🚀 Prochaines étapes :")
    print("   1. Exécutez main_multiclass.py pour le pipeline complet")
    print("   2. Ou consultez GUIDE_MULTICLASS.md pour plus de détails")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
