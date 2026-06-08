# Guide d'intégration : Classification Multi-classe avec CICIDS2017

## 📋 Vue d'ensemble

Ce guide explique comment utiliser les **nouvelles fonctions multi-classe** avec votre pipeline ML.

### ✨ Changements clés

| Ancien (Binaire) | Nouveau (Multi-classe) |
|------------------|----------------------|
| `load_dataset(csv_path)` | `load_dataset_multi_class(data_dir)` |
| `prepare_features_labels(df, ...)` | `prepare_features_labels_multiclass(df, ...)` |
| `split_data(X, y, ...)` | `split_data_multiclass(X, y, ...)` |
| Labels : `0` ou `1` | Labels : `"BENIGN"`, `"DDoS"`, `"Botnet"`, etc. |

---

## 🚀 Étape 1 : Charger tous les fichiers CSV par portions

### Code ancien (un seul fichier) :
```python
from src.data_loader import load_dataset
from config import DATA_PATH

df = load_dataset(DATA_PATH)  # Charge data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

### Code nouveau (tous les fichiers, par portions) :
```python
from pathlib import Path
from src.data_loader import load_dataset_multi_class

# Charge TOUS les fichiers .csv du dossier data/ 
# par portions de 50 000 lignes chaque
df = load_dataset_multi_class(
    data_dir=Path("data"),
    label_column="Label",
    chunk_size=50000,  # 50 000 lignes par fichier (adapter si besoin)
    random_state=42
)
```

**Avantages :**
- ✅ Charge **tous** les fichiers du dossier `data/`
- ✅ Fusion automatique via `pd.concat()`
- ✅ Limitation de RAM : 50 000 lignes par fichier max
- ✅ Labels **conservés en texte** (multi-classe)

---

## 🔄 Étape 2 : Préparation des features et labels (MULTI-CLASSE)

### Code ancien (conversion en binaire 0/1) :
```python
from src.preprocessing import prepare_features_labels

X, y = prepare_features_labels(
    df,
    label_column="Label",
    benign_label="BENIGN"  # 0 = BENIGN, 1 = attaque
)
```

### Code nouveau (conservation des labels textuels) :
```python
from src.preprocessing import prepare_features_labels_multiclass

X, y = prepare_features_labels_multiclass(
    df,
    label_column="Label"
    # AUCUN "benign_label" = labels restent en texte
)

# y est maintenant une Series contenant : "BENIGN", "DDoS", "Botnet", etc.
# Au lieu de : 0, 1, 1, 0, ...
```

**Affichage de la distribution :**
```
=== Préparation features/label (MULTI-CLASSE) ===
  Features : (50000, 79)
  Label    : (50000,)
  Nombre de classes : 5
  Distribution des classes :
    - BENIGN                 :        35000  ( 70.00%)
    - DDoS                   :        10000  ( 20.00%)
    - Botnet                 :         3000  (  6.00%)
    - Port Scan              :         1500  (  3.00%)
    - SSH Brute Force        :          500  (  1.00%)
```

---

## 📊 Étape 3 : Split stratifié (MULTI-CLASSE)

### Code ancien (split binaire) :
```python
from src.preprocessing import split_data

X_train, X_test, y_train, y_test = split_data(
    X, y, 
    test_size=0.2,
    random_state=42
)
```

### Code nouveau (split multi-classe) :
```python
from src.preprocessing import split_data_multiclass

X_train, X_test, y_train, y_test = split_data_multiclass(
    X, y,
    test_size=0.2,
    random_state=42
)
```

**Affichage :**
```
=== Split stratifié (80/20) - MULTI-CLASSE ===
  Train : 40000 samples
  Test  : 10000 samples

  Distribution classes (TRAIN) :
    - BENIGN                 :        28000  ( 70.00%)
    - DDoS                   :         8000  ( 20.00%)
    - Botnet                 :         2400  (  6.00%)
    - Port Scan              :         1200  (  3.00%)
    - SSH Brute Force        :          400  (  1.00%)

  Distribution classes (TEST) :
    - BENIGN                 :         7000  ( 70.00%)
    - DDoS                   :         2000  ( 20.00%)
    - Botnet                 :          600  (  6.00%)
    - Port Scan              :          300  (  3.00%)
    - SSH Brute Force        :          100  (  1.00%)
```

**Note importante :** `stratify=y` fonctionne parfaitement avec les labels textuels !
- ✅ Maintient l'équilibre des classes dans train/test
- ✅ Aucune conversion nécessaire

---

## 🧹 Étapes suivantes (sans changement)

Les autres étapes du pipeline restent **identiques** :

```python
from src.preprocessing import (
    clean_data,
    select_features,
    normalize_features,
)

# Nettoyage
df, stats_clean = clean_data(df)

# Sélection de features (fonctionne avec X_train multi-classe)
X_train_sel, X_test_sel, feature_importances, stats_feat = select_features(
    X_train, X_test, y_train,
    variance_threshold=0.0,
    correlation_threshold=0.95,
)

# Normalisation (fonctionne avec labels textuels)
X_train_norm, X_test_norm, scaler = normalize_features(X_train_sel, X_test_sel)
```

---

## 🤖 Entraînement des modèles

**Excellente nouvelle :** scikit-learn supporte **nativement** la classification multi-classe !

```python
from src.models import train_random_forest

# Le Random Forest gère automatiquement les labels textuels
model_rf = train_random_forest(
    X_train_norm, y_train,
    n_estimators=100,
    max_depth=20,
    n_jobs=-1,
)

# Génération de prédictions
y_pred_rf = model_rf.predict(X_test_norm)

# y_pred_rf contiendra : "BENIGN", "DDoS", "Botnet", etc.
# Au lieu de : 0, 1, 1, 0, ...
```

**Les modèles supportés :**
- ✅ Random Forest (parfait pour multi-classe)
- ✅ SVM (avec kernel linéaire ou RBF)
- ✅ MLP (réseau de neurones)

---

## 📈 Évaluation et métriques

Aucun changement dans `evaluation.py` ! Les métriques de scikit-learn supportent nativement multi-classe :

```python
from src.evaluation import evaluate_model, print_classification_report

metrics = evaluate_model(
    y_test, y_pred_rf,
    average="weighted"  # Moyenne pondérée pour multi-classe
)

print_classification_report(y_test, y_pred_rf)
```

**Affichage multi-classe :**
```
              precision    recall  f1-score   support

      BENIGN       0.96      0.97      0.96      7000
        DDoS       0.92      0.90      0.91      2000
      Botnet       0.85      0.82      0.83       600
   Port Scan       0.78      0.75      0.76       300
SSH Brute Force    0.70      0.65      0.67       100

    accuracy                           0.93     10000
   macro avg       0.84      0.82      0.83     10000
weighted avg       0.93      0.93      0.93     10000
```

---

## ⚙️ Configuration (config.py)

**Aucune modification majeure !** Mais vous pouvez ajouter :

```python
# config.py

# Pour le chargement multi-classe
DATA_DIR = ROOT_DIR / "data"  # Déjà présent
CHUNK_SIZE = 50000  # Nouvelle : taille des portions

# Pour la classification multi-classe
# Ancien paramètre (plus utilisé)
BENIGN_LABEL = "BENIGN"  # Gardé pour compatibilité, mais inutile

# Nouveau paramètre (optionnel)
LABEL_COLUMN = "Label"  # Nom de la colonne label
```

---

## 📝 Exemple complet : Pipeline multi-classe

```python
"""
Pipeline ML multi-classe avec CICIDS2017.
"""
from pathlib import Path
from src.data_loader import load_dataset_multi_class, describe_dataset, print_dataset_summary
from src.preprocessing import (
    clean_data,
    prepare_features_labels_multiclass,
    split_data_multiclass,
    select_features,
    normalize_features,
)
from src.models import train_random_forest, get_predictions
from src.evaluation import evaluate_model, print_classification_report
import config

# 1. Chargement MULTI-CLASSE (tous les fichiers .csv, par portions)
print("ÉTAPE 1 : Chargement multi-classe")
df = load_dataset_multi_class(
    data_dir=config.DATA_DIR,
    label_column=config.LABEL_COLUMN,
    chunk_size=50000,
    random_state=config.RANDOM_STATE,
)

stats = describe_dataset(df, config.LABEL_COLUMN)
print_dataset_summary(stats)

# 2. Nettoyage
print("\nÉTAPE 2 : Nettoyage")
df, _ = clean_data(df)

# 3. Préparation MULTI-CLASSE
print("\nÉTAPE 3 : Préparation features/label (multi-classe)")
X, y = prepare_features_labels_multiclass(
    df,
    label_column=config.LABEL_COLUMN
)

# 4. Split MULTI-CLASSE
print("\nÉTAPE 4 : Split stratifié (multi-classe)")
X_train, X_test, y_train, y_test = split_data_multiclass(
    X, y,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE,
)

# 5. Sélection de features
print("\nÉTAPE 5 : Sélection de features")
X_train_sel, X_test_sel, feat_imp, _ = select_features(
    X_train, X_test, y_train,
    correlation_threshold=config.CORRELATION_THRESHOLD,
)

# 6. Normalisation
print("\nÉTAPE 6 : Normalisation")
X_train_norm, X_test_norm, scaler = normalize_features(X_train_sel, X_test_sel)

# 7. Entraînement
print("\nÉTAPE 7 : Entraînement Random Forest (multi-classe)")
model = train_random_forest(
    X_train_norm, y_train,
    n_estimators=100,
    max_depth=20,
    n_jobs=-1,
)

# 8. Prédictions
print("\nÉTAPE 8 : Prédictions")
y_pred = get_predictions(model, X_test_norm)

# 9. Évaluation
print("\nÉTAPE 9 : Évaluation")
metrics = evaluate_model(y_test, y_pred, average="weighted")
print_classification_report(y_test, y_pred)

print("\n✅ Pipeline multi-classe complété !")
```

---

## 🔑 Points clés à retenir

| ✅ À faire | ❌ À éviter |
|-----------|-----------|
| Utiliser `load_dataset_multi_class()` pour charger plusieurs fichiers | Charger un seul fichier à la fois |
| Utiliser `prepare_features_labels_multiclass()` pour garder les labels textuels | Convertir les labels en binaire 0/1 |
| Utiliser `split_data_multiclass()` avec labels textuels | Passer des labels binaires à split_data |
| Utiliser `stratify=y` (déjà inclus) | Oublier de stratifier le split |
| Utiliser `average="weighted"` dans les métriques | Utiliser `average="binary"` |

---

## ❓ FAQ

**Q: Dois-je modifier les fichiers d'entraînement (models.py) ?**  
A: Non ! Scikit-learn gère nativement multi-classe. Les labels textuels sont acceptés directement.

**Q: Comment gérer les déséquilibres de classes ?**  
A: Utilisez `class_weight="balanced"` dans les modèles :
```python
model_rf = RandomForestClassifier(class_weight="balanced", n_estimators=100)
```

**Q: Puis-je encore utiliser la classification binaire ?**  
A: Oui ! Les anciennes fonctions (`prepare_features_labels`, `split_data`) restent disponibles.

**Q: Comment sauvegarde/charger le scaler ?**  
A: Identique à avant (fonctionne avec labels textuels) :
```python
import joblib
joblib.dump(scaler, config.MODELS_DIR / "scaler.joblib")
scaler_loaded = joblib.load(config.MODELS_DIR / "scaler.joblib")
```

---

## 📚 Fichiers modifiés

- **data_loader.py** : Ajout de `load_dataset_multi_class()`
- **preprocessing.py** : Ajout de `prepare_features_labels_multiclass()` et `split_data_multiclass()`
- **Aucun changement** : models.py, evaluation.py, visualization.py ✅

---

## 🎯 Prochaines étapes

1. Testez avec votre dataset CICIDS2017
2. Comparez les performances multi-classe vs binaire
3. Ajustez `chunk_size` selon votre RAM disponible
4. Utilisez `class_weight="balanced"` si déséquilibre détecté

Bon travail ! 🚀
