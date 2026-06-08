"""
Configuration centralisée du projet.

Tous les chemins, seuils et hyperparamètres sont regroupés ici pour faciliter
la maintenance et l'expérimentation.
"""
from pathlib import Path

# ============================================================
# Chemins du projet
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"

# Création automatique des dossiers de sortie
for d in (FIGURES_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Dataset
# ============================================================
# Nom du fichier CSV à analyser (à placer dans data/)
DATA_PATH = Path(r"data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
DATA_FILENAME = DATA_PATH.name

# Nom de la colonne contenant le label
LABEL_COLUMN = "Label"

# Valeur du label correspondant au trafic normal (le reste est considéré comme attaque)
BENIGN_LABEL = "BENIGN"

# ============================================================
# Reproductibilité
# ============================================================
RANDOM_STATE = 42

# ============================================================
# Préparation des données
# ============================================================
# Proportion de l'ensemble de test
TEST_SIZE = 0.20

# Seuil de corrélation pour la sélection de features
# Toute paire de features avec |corr| > ce seuil voit l'une des deux retirée
CORRELATION_THRESHOLD = 0.95

# Seuil de variance pour le VarianceThreshold (0 = retirer seulement les constantes)
VARIANCE_THRESHOLD = 0.0

# ============================================================
# Hyperparamètres des modèles
# ============================================================
# Random Forest
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Random Forest "léger" utilisé pour calculer l'importance des features
RF_SELECTOR_PARAMS = {
    "n_estimators": 50,
    "max_depth": 15,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# SVM linéaire
SVM_LINEAR_PARAMS = {
    "random_state": RANDOM_STATE,
    "max_iter": 5000,
    "dual": False,   # primale, plus rapide quand n_samples >> n_features
}

# SVM à noyau RBF (entraîné sur un sous-échantillon)
SVM_RBF_PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "random_state": RANDOM_STATE,
    "probability": True,
}
# Taille de l'échantillon utilisé pour le SVM RBF (pour limiter le temps)
SVM_RBF_SAMPLE_SIZE = 20000

# MLP (réseau de neurones simple)
MLP_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "learning_rate": "adaptive",
    "max_iter": 100,
    "random_state": RANDOM_STATE,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
}

# ============================================================
# Visualisation
# ============================================================
FIGURE_DPI = 150
FIGURE_FORMAT = "png"
