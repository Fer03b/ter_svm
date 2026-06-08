# Pipeline de Classification Multi-Classe — TER M1 RSA

Pipeline complet du projet TER — **Classification MULTI-CLASSE** avec CICIDS2017.

## Description

Ce script implémenter un pipeline complet de classification multi-classe pour la détection d'attaques réseau. 
Les labels sont conservés en texte (ex: "BENIGN", "DDoS", "Botnet") au lieu d'être convertis en binaire.

## Étapes du pipeline

1. **Chargement et exploration** — Charge tous les fichiers CSV du dossier `data/` par portions de 50 000 lignes
2. **Nettoyage** — Suppression des NaN, Inf, doublons
3. **Préparation multi-classe** — Prépare features/labels en gardant les labels textuels
4. **Split stratifié** — Split train/test stratifié adapté au multi-classe
5. **Sélection de features** — Filtrage par variance et corrélation
6. **Normalisation** — Normalisation des features
7. **Entraînement des modèles** — Quatre modèles entraînés nativement pour le multi-classe
8. **Évaluation** — Évaluation avec `average="weighted"` pour le multi-classe
9. **Visualisations** — Génération des graphiques et sauvegarde des résultats

## Modèles entraînés

- **Random Forest** (100 estimateurs, max_depth=20)
- **SVM Linéaire** (kernel linéaire, C=1.0, max_iter=1000)
- **SVM RBF** (kernel RBF, C=1.0, gamma='scale')
- **MLP** (2 couches cachées 128-64, solver='adam', max_iter=500)

## Utilisation

```bash
python main_multiclass.py
```

## Résultats générés

### Modèles
- `outputs/models/random_forest_multiclass.joblib`
- `outputs/models/svm_linear_multiclass.joblib`
- `outputs/models/svm_rbf_multiclass.joblib`
- `outputs/models/mlp_multiclass.joblib`
- `outputs/models/scaler_multiclass.joblib`

### Figures
- `outputs/figures/01_class_distribution_multiclass.png` — Distribution des classes
- `outputs/figures/confusion_matrices_multiclass.png` — Matrices de confusion (4 modèles)
- `outputs/figures/metrics_comparison_multiclass.png` — Comparaison des métriques
- `outputs/figures/feature_importance_multiclass.png` — Top 20 features (Random Forest)

### Résultats
- `outputs/results/comparison_table_multiclass.csv` — Tableau comparatif des modèles

## Métriques évaluées

Pour chaque modèle (average="weighted") :
- Accuracy
- Précision
- Rappel (Recall)
- F1-score

## Configuration

Modifiez `config.py` pour ajuster :
- `DATA_DIR` — Dossier contenant les fichiers CSV
- `LABEL_COLUMN` — Colonne des labels (par défaut: "Label")
- `TEST_SIZE` — Proportion du test set (par défaut: 0.2)
- `CORRELATION_THRESHOLD` — Seuil de corrélation pour la sélection de features
- `RANDOM_STATE` — Graine aléatoire pour la reproductibilité
