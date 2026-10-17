
from pathlib import Path

import pandas as pd


def validate_dataset(
    dataset_path: str,
    target_column: str,
    task: str = "auto",
) -> dict:
    """Validate the dataset before feature engineering."""

    path = Path(dataset_path)

    if not path.exists():
        return {
            "valid": False,
            "message": f"Dataset not found: {path}",
        }

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return {
            "valid": False,
            "message": f"Could not read CSV: {exc}",
        }

    if df.empty:
        return {
            "valid": False,
            "message": "The dataset is empty.",
        }

    if not df.columns.is_unique:
        return {
            "valid": False,
            "message": "The dataset contains duplicate column names.",
        }

    if target_column not in df.columns:
        return {
            "valid": False,
            "message": (
                f"Target column '{target_column}' was not found. "
                f"Available columns: {list(df.columns)}"
            ),
        }

    # Rows without a target cannot be used for supervised training.
    valid_target = df[target_column].dropna()

    if len(valid_target) < 2:
        return {
            "valid": False,
            "message": "At least two rows with a target value are required.",
        }

    selected_task = task.lower()

    if selected_task not in {"auto", "classification", "regression"}:
        return {
            "valid": False,
            "message": (
                "Task must be auto, classification, or regression."
            ),
        }

    if selected_task == "auto":
        if (
            pd.api.types.is_object_dtype(valid_target)
            or pd.api.types.is_bool_dtype(valid_target)
            or isinstance(valid_target.dtype, pd.CategoricalDtype)
        ):
            selected_task = "classification"
        else:
            selected_task = "regression"

    if selected_task == "classification":
        if valid_target.nunique() < 2:
            return {
                "valid": False,
                "message": (
                    "Classification requires at least two target classes."
                ),
            }

    elif selected_task == "regression":
        numeric_target = pd.to_numeric(valid_target, errors="coerce")

        if numeric_target.isna().any():
            return {
                "valid": False,
                "message": (
                    "Regression requires a numeric target column."
                ),
            }

    return {
        "valid": True,
        "message": "Dataset validation passed.",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "target_column": target_column,
        "task": selected_task,
    }
