import pytest
import pandas as pd
import numpy as np
from src.data.data_validator import validate_dataset

def test_validate_dataset_success():
    df = pd.DataFrame({
        "overall": [80, 90, 85],
        "pace": [85, 95, 90],
        "name": ["A", "B", "C"]
    })

    report = validate_dataset(df)
    assert report["rows"] == 3
    assert report["columns"] == 3
    assert report["target_exists"] is True
    assert report["duplicate_rows"] == 0
    assert len(report["missing_values"]) == 0
    assert "pace" in report["numerical_columns"]
    assert "name" in report["categorical_columns"]
    assert len(report["target_issues"]) == 0

def test_validate_dataset_missing_target():
    df = pd.DataFrame({
        "pace": [85, 95],
        "name": ["A", "B"]
    })
    report = validate_dataset(df)
    assert report["target_exists"] is False

def test_validate_dataset_missing_values():
    df = pd.DataFrame({
        "overall": [80, 90, np.nan],
        "pace": [85, np.nan, 90]
    })
    report = validate_dataset(df)
    assert report["missing_values"]["pace"] == 1
    assert "Target column contains missing values." in report["target_issues"]

def test_validate_dataset_duplicates():
    df = pd.DataFrame({
        "overall": [80, 80],
        "pace": [85, 85]
    })
    report = validate_dataset(df)
    assert report["duplicate_rows"] == 1

def test_validate_dataset_invalid_input():
    with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
        validate_dataset([1, 2, 3])

def test_validate_dataset_target_issues():
    df = pd.DataFrame({
        "overall": [80, 150, -10]
    })
    report = validate_dataset(df)
    assert any("outside [0, 100]" in issue for issue in report["target_issues"])

    df_str = pd.DataFrame({
        "overall": ["80", "90"]
    })
    report_str = validate_dataset(df_str)
    assert any("not numeric" in issue for issue in report_str["target_issues"])
