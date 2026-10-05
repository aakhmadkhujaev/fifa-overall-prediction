import pandas as pd
from typing import Dict, Any

def validate_dataset(df: object) -> Dict[str, Any]:
    """
    Validates the dataset structure and quality without modifying it.

    Args:
        df (pd.DataFrame): Dataset to validate.

    Returns:
        dict: A structured validation report.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)}")

    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "target_exists": False,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {},
        "numerical_columns": [],
        "categorical_columns": [],
        "target_issues": []
    }

    # Target check
    if "overall" in df.columns:
        report["target_exists"] = True
        target = df["overall"]
        if target.isnull().any():
            report["target_issues"].append("Target column contains missing values.")
        if not pd.api.types.is_numeric_dtype(target):
            report["target_issues"].append("Target column is not numeric.")
        else:
            if (target < 0).any() or (target > 100).any():
                report["target_issues"].append("Target column contains values outside [0, 100].")

    # Missing values per column
    missing = df.isnull().sum()
    report["missing_values"] = missing[missing > 0].to_dict()

    # Column types
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            report["numerical_columns"].append(col)
        else:
            report["categorical_columns"].append(col)

    return report
