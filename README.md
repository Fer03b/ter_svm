# TER M1 RSA — Détection d'attaques réseau par Machine Learning

Système de Détection d'Intrusion (IDS) basé sur des techniques de Machine Learning,
appliqué au dataset CICIDS2017.

## Objectifs (extrait du sujet)

Concevoir et évaluer un système de détection d'intrusion basé sur le Machine Learning,
puis comparer ses performances avec une approche IDS traditionnelle.

## Modèles implémentés

- **Random Forest** (méthode d'ensemble)
- **SVM** — LinearSVC et SVC avec noyau RBF (méthode à noyau)
- **MLP** (réseau de neurones simple, 2 couches cachées)

## Structure du projet

```
ter_ids_ml/
├── README.md             # Ce fichier
├── requirements.txt      # Dépendances Python
├── config.py             # Configuration centrale (chemins, hyperparamètres)
├── main.py               # Pipeline complet
│
├── data/                 # Fichiers CSV de CICIDS2017 à placer ici
│
├── outputs/              # Résultats générés (créé automatiquement)
│   ├── figures/          # Graphiques (.png)
│   ├── models/           # Modèles entraînés (.joblib)
│   └── results/          # Tableaux de résultats (.csv)
│
└── src/                  # Code source modulaire
    ├── data_loader.py    # Chargement du dataset
    ├── preprocessing.py  # Nettoyage, sélection, split, normalisation
    ├── models.py         # Entraînement des modèles
    ├── evaluation.py     # Métriques (accuracy, F1, taux FP, etc.)
    └── visualization.py  # Graphiques (ROC, matrices de confusion, etc.)
```

## Installation

### Prérequis
- Python 3.10+
- pip

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Téléchargement du dataset

1. Téléchargez CICIDS2017 (version MachineLearningCSV) sur :
   https://www.unb.ca/cic/datasets/ids-2017.html
2. Placez le fichier CSV dans le dossier `data/`
3. Le fichier par défaut attendu est :
   `data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
4. Pour utiliser un autre fichier, modifiez `DATA_FILENAME` dans `config.py`

## Utilisation

### Pipeline complet (recommandé)

```bash
python main.py
```

Cela lance l'intégralité du pipeline :
1. Chargement et exploration des données
2. Nettoyage (suppression des NaN, Inf, doublons)
3. Sélection des features (variance, corrélation, importance)
4. Split train/test stratifié + normalisation
5. Entraînement des 4 modèles (RF, SVM linéaire, SVM RBF, MLP)
6. Évaluation et comparaison
7. Génération des figures et sauvegarde des résultats

### Résultats produits

- **`outputs/figures/`** : graphiques (distribution des classes, importance des features,
  matrice de corrélation, courbes ROC, matrices de confusion, comparaison des métriques)
- **`outputs/models/`** : modèles entraînés au format joblib (rechargeables sans réentraînement)
- **`outputs/results/`** : tableau de comparaison des modèles et importance des features (CSV)

## Métriques évaluées

Conformément au sujet :
- Accuracy
- Précision
- Rappel (Recall)
- F1-score
- **Taux de faux positifs (FPR)**
- ROC-AUC
- Temps d'entraînement

## Auteur

Projet TER M1 RSA — Année universitaire 2025-2026
