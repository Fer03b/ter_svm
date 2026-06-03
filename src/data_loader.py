"""
Module de chargement du dataset CICIDS2017.

Le chargement inclut le nettoyage des noms de colonnes (CICIDS2017 contient
souvent des espaces parasites dans les en-têtes).
"""
from pathlib import Path
import pandas as pd


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Charge un fichier CSV de CICIDS2017.

    Parameters
    ----------
    csv_path : Path
        Chemin vers le fichier CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame avec les noms de colonnes nettoyés (sans espaces parasites).

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {csv_path}\n"
            f"Téléchargez CICIDS2017 sur https://www.unb.ca/cic/datasets/ids-2017.html "
            f"et placez le CSV dans le dossier data/."
        )

    print(f"Chargement de {csv_path.name}...")
    df = pd.read_csv(csv_path)

    # Nettoyage des espaces parasites dans les noms de colonnes
    df.columns = df.columns.str.strip()

    print(f"  → {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    return df


def describe_dataset(df: pd.DataFrame, label_column: str = "Label") -> dict:
    """
    Calcule des statistiques descriptives sur le dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Le dataset.
    label_column : str
        Nom de la colonne contenant le label.

    Returns
    -------
    dict
        Dictionnaire contenant les statistiques principales.
    """
    import numpy as np

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    stats = {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "n_features": df.shape[1] - 1,
        "class_distribution": df[label_column].value_counts().to_dict(),
        "class_proportions": (df[label_column].value_counts(normalize=True) * 100).round(2).to_dict(),
        "n_nan": int(df.isnull().sum().sum()),
        "n_inf": int(np.isinf(df[numeric_cols]).sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }

    return stats


def print_dataset_summary(stats: dict) -> None:
    """Affiche un résumé lisible des statistiques du dataset."""
    print("\n=== Résumé du dataset ===")
    print(f"  Lignes      : {stats['n_rows']:,}")
    print(f"  Features    : {stats['n_features']}")
    print(f"  Mémoire     : {stats['memory_mb']:.2f} Mo")
    print(f"  NaN         : {stats['n_nan']:,}")
    print(f"  Inf         : {stats['n_inf']:,}")
    print(f"  Doublons    : {stats['n_duplicates']:,}")
    print(f"  Distribution des classes :")
    for label, count in stats["class_distribution"].items():
        pct = stats["class_proportions"][label]
        print(f"    - {label:<15} : {count:>10,}  ({pct:.2f}%)")
