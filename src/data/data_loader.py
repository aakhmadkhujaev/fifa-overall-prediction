import pandas as pd
from pathlib import Path
from typing import Union, Optional

class DataLoaderError(Exception):
    """Custom exception for errors during data loading."""
    pass

def load_dataset(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load the FIFA dataset from a CSV file.

    Args:
        file_path (str or Path, optional): Path to the dataset.
            Defaults to 'data/raw/male_players (legacy).csv' relative to project root.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        DataLoaderError: If the file does not exist or cannot be parsed.
    """
    if file_path is None:
        root_dir = Path(__file__).resolve().parent.parent.parent
        file_path = root_dir / "data" / "raw" / "male_players (legacy).csv"

    file_path = Path(file_path)

    if not file_path.exists():
        raise DataLoaderError(f"Dataset file not found at: {file_path}")

    if not file_path.is_file():
        raise DataLoaderError(f"Path is not a file: {file_path}")

    try:
        df = pd.read_csv(file_path, low_memory=False)
        return df
    except Exception as e:
        raise DataLoaderError(f"Failed to load dataset: {e}") from e
