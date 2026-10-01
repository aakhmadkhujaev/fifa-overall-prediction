import pytest
import pandas as pd
from pathlib import Path
from src.data.data_loader import load_dataset, DataLoaderError

def test_load_dataset_success(tmp_path):
    csv_file = tmp_path / "dummy.csv"
    df = pd.DataFrame({"overall": [80, 90], "pace": [85, 95]})
    df.to_csv(csv_file, index=False)

    loaded_df = load_dataset(csv_file)
    assert isinstance(loaded_df, pd.DataFrame)
    assert len(loaded_df) == 2
    assert list(loaded_df.columns) == ["overall", "pace"]

def test_load_dataset_missing_file(tmp_path):
    missing_file = tmp_path / "nonexistent.csv"
    with pytest.raises(DataLoaderError, match="Dataset file not found"):
        load_dataset(missing_file)

def test_load_dataset_invalid_file_type(tmp_path):
    dir_path = tmp_path / "is_dir"
    dir_path.mkdir()
    with pytest.raises(DataLoaderError, match="Path is not a file"):
        load_dataset(dir_path)

def test_load_dataset_parsing_error(tmp_path):
    # Using a file that isn't valid csv but we force a parsing error by using a different approach
    invalid_file = tmp_path / "bad.csv"
    # Just to simulate an unreadable file or parsing error:
    with open(invalid_file, 'wb') as f:
        f.write(b'\x00\xFF\xFE\x00\xFF')

    with pytest.raises(DataLoaderError, match="Failed to load dataset"):
        load_dataset(invalid_file)
