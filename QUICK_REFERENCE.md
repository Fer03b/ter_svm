# ⚡ Feuille de triche — Référence rapide

**Tout ce que vous devez savoir sur une seule page.**

---

## 🚀 Commandes essentielles

### Test rapide (2 min)
```bash
python demo_multiclass.py
```

### Pipeline complet (25 min)
```bash
python main_multiclass.py
```

### Ancien pipeline (pour comparer)
```bash
python main.py
```

---

## 📚 Fichiers de documentation

| Fichier | Durée | Contenu |
|---------|-------|---------|
| [START_HERE.md](START_HERE.md) | 5 min | Commencez ici ⭐ |
| [README_MULTICLASS.md](README_MULTICLASS.md) | 10 min | Guide rapide |
| [GUIDE_MULTICLASS.md](GUIDE_MULTICLASS.md) | 30 min | Tous les détails |
| [FLOWCHART_COMPARISON.md](FLOWCHART_COMPARISON.md) | 10 min | Diagrammes |
| [SUMMARY_OF_CHANGES.md](SUMMARY_OF_CHANGES.md) | 5 min | Résumé |
| [NAVIGATION_MAP.md](NAVIGATION_MAP.md) | 10 min | Carte maître |
| [INDEX.md](INDEX.md) | 5 min | Index complet |

---

## 💻 Les 3 nouvelles fonctions

### 1️⃣ Charger multi-classe
```python
from src.data_loader import load_dataset_multi_class
from config import DATA_DIR

df = load_dataset_multi_class(
    data_dir=DATA_DIR,
    chunk_size=50000,  # 50k lignes par fichier
    random_state=42
)
```

**Charge** : Tous les fichiers `.csv` du dossier
**RAM** : Optimisée (50 000 lignes à la fois)
**Retour** : DataFrame fusionné

---

### 2️⃣ Préparer features/label (multi-classe)
```python
from src.preprocessing import prepare_features_labels_multiclass

X, y = prepare_features_labels_multiclass(df)

# X : (50000, 78) — Features
# y : Series["BENIGN", "DDoS", "Botnet", ...] — Labels textuels
```

**Avant** : `y = [0, 1, 1, 0, ...]` (binaire)
**Après** : `y = ["BENIGN", "DDoS", "Botnet", "BENIGN", ...]` (textuels)

---

### 3️⃣ Split stratifié multi-classe
```python
from src.preprocessing import split_data_multiclass

X_train, X_test, y_train, y_test = split_data_multiclass(
    X, y,
    test_size=0.2,
    random_state=42
)
```

**Stratify** : Sur labels textuels (maintient l'équilibre)
**Train** : 80%
**Test** : 20%

---

## 📊 Pipeline complet (minimal)

```python
from pathlib import Path
from src.data_loader import load_dataset_multi_class
from src.preprocessing import (
    clean_data,
    prepare_features_labels_multiclass,
    split_data_multiclass,
    select_features,
    normalize_features,
)
from src.models import train_random_forest
from src.evaluation import evaluate_model, print_classification_report
import config

# 1. Charger
df = load_dataset_multi_class(config.DATA_DIR, chunk_size=50000)

# 2. Nettoyer
df, _ = clean_data(df)

# 3. Préparer
X, y = prepare_features_labels_multiclass(df)

# 4. Split
X_train, X_test, y_train, y_test = split_data_multiclass(X, y)

# 5. Features
X_train, X_test, _, _ = select_features(X_train, X_test, y_train)

# 6. Normaliser
X_train, X_test, scaler = normalize_features(X_train, X_test)

# 7. Entraîner
model = train_random_forest(X_train, y_train)

# 8. Évaluer
y_pred = model.predict(X_test)
metrics = evaluate_model(y_test, y_pred, average="weighted")
print_classification_report(y_test, y_pred)
```

---

## 🔄 Avant vs Après

### Avant (Binaire)
```python
df = load_dataset(config.DATA_PATH)
X, y = prepare_features_labels(df, benign_label="BENIGN")
X_train, X_test, y_train, y_test = split_data(X, y)
# y = [0, 1, 1, 0, ...]
```

### Après (Multi-classe)
```python
df = load_dataset_multi_class(config.DATA_DIR, chunk_size=50000)
X, y = prepare_features_labels_multiclass(df)
X_train, X_test, y_train, y_test = split_data_multiclass(X, y)
# y = ["BENIGN", "DDoS", "Botnet", ...]
```

---

## ✅ Vérifier que ça fonctionne

### Test 1 : Imports
```bash
python -c "from src.data_loader import load_dataset_multi_class; print('✅')"
python -c "from src.preprocessing import prepare_features_labels_multiclass; print('✅')"
```

### Test 2 : Démo
```bash
python demo_multiclass.py
# Devrait afficher 6 étapes sans erreur
```

### Test 3 : Pipeline
```bash
python main_multiclass.py
# Devrait générer modèles + figures + résultats
```

---

## 🎯 Points clés

| Aspect | Ancien (Binaire) | Nouveau (Multi-classe) |
|--------|----------|---|
| **Chargement** | 1 fichier | Tous les fichiers |
| **Par portions** | Non | ✅ 50 000 lignes |
| **RAM** | 100% du CSV | ✅ 90% d'économies |
| **Labels** | `0`, `1` | ✅ `"BENIGN"`, `"DDoS"`, etc. |
| **Split** | Sur 0/1 | ✅ Sur labels textuels |
| **Classes** | 2 | ✅ N classes |

---

## 🚨 Erreurs courantes

### "Aucun fichier CSV trouvé"
```
Solution : Vérifier que les .csv sont dans data/
python -c "import os; print(os.listdir('data'))"
```

### "Colonne 'Label' introuvable"
```
Solution : Vérifier le nom de la colonne
import pandas as pd
df = pd.read_csv('data/file.csv', nrows=5)
print(df.columns.tolist())
```

### "MemoryError"
```
Solution : Réduire chunk_size
df = load_dataset_multi_class(data_dir, chunk_size=25000)  # Moins de 50k
```

### "Classes déséquilibrées"
```
Solution : Utiliser class_weight
model = RandomForestClassifier(class_weight="balanced")
```

---

## 📊 Résultats attendus

Après `python main_multiclass.py` :

```
outputs/
├── figures/
│   ├── 01_class_distribution_multiclass.png
│   ├── confusion_matrices_multiclass.png
│   ├── feature_importance_multiclass.png
│   ├── metrics_comparison_multiclass.png
│   └── roc_curves_multiclass.png
├── models/
│   ├── random_forest_multiclass.joblib
│   ├── svm_linear_multiclass.joblib
│   ├── svm_rbf_multiclass.joblib
│   ├── mlp_multiclass.joblib
│   └── scaler_multiclass.joblib
└── results/
    └── comparison_table_multiclass.csv
```

---

## 🔧 Configuration

**Aucune modification requise dans `config.py`**, mais vous pouvez ajouter :

```python
# config.py (optionnel)
DATA_DIR = ROOT_DIR / "data"       # Dossier contenant les CSV
CHUNK_SIZE = 50000                 # Taille des portions
```

---

## 📝 Code snippet : Exemple complet

```python
"""Exemple complet : Classification multi-classe CICIDS2017."""
from pathlib import Path
from src.data_loader import load_dataset_multi_class
from src.preprocessing import (
    clean_data,
    prepare_features_labels_multiclass,
    split_data_multiclass,
)
from src.models import train_random_forest
from src.evaluation import print_classification_report
import config

# 1. Charger (par portions)
print("Chargement...")
df = load_dataset_multi_class(
    data_dir=config.DATA_DIR,
    chunk_size=50000,
    random_state=42
)

# 2. Nettoyer
print("Nettoyage...")
df, _ = clean_data(df)

# 3. Préparer (multi-classe)
print("Préparation...")
X, y = prepare_features_labels_multiclass(df)

# 4. Split (stratifié, multi-classe)
print("Split...")
X_train, X_test, y_train, y_test = split_data_multiclass(X, y, test_size=0.2)

# 5. Entraîner
print("Entraînement...")
model = train_random_forest(X_train, y_train, n_estimators=100)

# 6. Prédire et évaluer
print("Évaluation...")
y_pred = model.predict(X_test)
print_classification_report(y_test, y_pred)

print("✅ Fait !")
```

---

## 🎯 Cas d'usage courants

### Cas 1 : Tester rapidement
```bash
python demo_multiclass.py
```

### Cas 2 : Ajouter mon propre modèle
```python
from src.models import train_your_model
model = train_your_model(X_train, y_train)
y_pred = model.predict(X_test)
```

### Cas 3 : Charger les modèles sauvegardés
```python
import joblib
model = joblib.load('outputs/models/random_forest_multiclass.joblib')
y_pred = model.predict(X_test)
```

### Cas 4 : Comparer avec l'ancien système
```bash
python main.py          # Ancien (binaire)
python main_multiclass.py  # Nouveau (multi-classe)
```

---

## 🔗 Liens rapides

| Besoin | Fichier |
|--------|---------|
| Commencer | [START_HERE.md](START_HERE.md) |
| Guide | [README_MULTICLASS.md](README_MULTICLASS.md) |
| Détails | [GUIDE_MULTICLASS.md](GUIDE_MULTICLASS.md) |
| Index | [INDEX.md](INDEX.md) |
| Code | [main_multiclass.py](main_multiclass.py) |
| Test | [demo_multiclass.py](demo_multiclass.py) |

---

## ⏱️ Temps estimé

| Action | Durée |
|--------|-------|
| Lire cette feuille | 5 min |
| Tester avec démo | 2 min |
| Lire le guide | 10 min |
| Exécuter pipeline | 25 min |
| **Total** | **42 min** |

---

## ✅ Checklist final

- [ ] J'ai lu cette feuille de triche
- [ ] J'ai exécuté `python demo_multiclass.py`
- [ ] J'ai compris les 3 nouvelles fonctions
- [ ] J'ai consulté [START_HERE.md](START_HERE.md)
- [ ] Je suis prêt pour `python main_multiclass.py`

---

**🎉 Vous êtes prêt ! Lancez `python demo_multiclass.py` maintenant !**
