"""
Module de chargement du dataset CICIDS2017 (Multi-classe avec chargement par portions).

Couvre le chargement de multiples fichiers CSV avec:
  - Lecture par portions pour limiter la consommation RAM
  - Fusion des données de tous les fichiers du dossier data/
  - Nettoyage des noms de colonnes (CICIDS2017 contient souvent des espaces parasites)
  - Conservation des labels textuels pour classification multi-classe
"""
from pathlib import Path
import pandas as pd


def load_dataset_multi_class(
    data_dir: Path,
    label_column: str = "Label",
    chunk_size: int = 50000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Charge TOUS les fichiers CSV du dossier data/ par portions, les fusionne,
    et conserve les labels textuels pour classification multi-classe.

    Cette fonction charge chaque fichier CSV par chunks de taille définie
    (par défaut 50 000 lignes) pour éviter de saturer la RAM sur un Mac.
    Les portions de chaque fichier sont puis fusionnées en un seul DataFrame.

    Parameters
    ----------
    data_dir : Path
        Chemin vers le dossier contenant les fichiers CSV (ex: data/).
    label_column : str
        Nom de la colonne contenant le label.
    chunk_size : int
        Taille des portions à charger par fichier (50 000 par défaut).
    random_state : int
        Graine aléatoire pour reproductibilité.

    Returns
    -------
    pd.DataFrame
        DataFrame fusionné avec tous les fichiers, colonnes nettoyées,
        labels conservés en texte.

    Raises
    ------
    FileNotFoundError
        Si aucun fichier CSV n'existe dans le dossier.
    """
    # Récupération de tous les fichiers CSV du dossier
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"Aucun fichier CSV trouvé dans {data_dir}\n"
            f"Téléchargez CICIDS2017 sur https://www.unb.ca/cic/datasets/ids-2017.html "
            f"et placez les CSV dans le dossier data/."
        )

    all_chunks = []
    total_rows_loaded = 0

    print(f"\n=== Chargement multi-classe (par portions de {chunk_size:,} lignes) ===")
    print(f"  Fichiers trouvés : {len(csv_files)}\n")

    # Parcourir chaque fichier CSV
    for csv_file in csv_files:
        print(f"  Chargement : {csv_file.name}")
        try:
            # Charger le fichier par chunks
            df_chunks = pd.read_csv(
                csv_file,
                nrows=chunk_size,  # Limitation à 50 000 lignes par fichier
            )

            # Nettoyage des espaces parasites dans les noms de colonnes
            df_chunks.columns = df_chunks.columns.str.strip()

            # Vérifier la présence de la colonne label
            if label_column not in df_chunks.columns:
                print(f"    ⚠ Colonne '{label_column}' introuvable. Colonnes disponibles: {list(df_chunks.columns)}")
                continue

            n_rows = df_chunks.shape[0]
            n_cols = df_chunks.shape[1]
            all_chunks.append(df_chunks)
            total_rows_loaded += n_rows

            # Afficher les classes présentes dans ce fichier
            label_distribution = df_chunks[label_column].value_counts().to_dict()
            print(f"    → {n_rows:,} lignes × {n_cols} colonnes | Classes: {label_distribution}")

        except Exception as e:
            print(f"    ✗ Erreur lors du chargement : {e}")
            continue

    # Fusion de tous les chunks en un seul DataFrame
    if not all_chunks:
        raise ValueError("Aucune donnée n'a pu être chargée.")

    df_combined = pd.concat(all_chunks, axis=0, ignore_index=True)

    print(f"\n  ✓ Fusion complétée : {total_rows_loaded:,} lignes chargées")
    print(f"  ✓ Shape final : {df_combined.shape[0]:,} lignes × {df_combined.shape[1]} colonnes")

    return df_combined


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    ⚠ DEPRECATED : Utilisez load_dataset_multi_class() pour la classification multi-classe.
    
    Charge un fichier CSV unique de CICIDS2017.
    Conservé pour compatibilité avec ancien code.

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
