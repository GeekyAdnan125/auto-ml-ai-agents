
from pathlib import Path

import pandas as pd

from tools.model_tools import detect_task, train_and_compare


BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"


def train_models(
    target_column: str,
    task: str = "auto",
) -> dict:
    """Train and compare models using Level 3 outputs."""

    train_path = REPORT_DIR / "prepared_train.csv"
    test_path = REPORT_DIR / "prepared_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Prepared datasets not found. Run Level 3 first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target_column not in train_df.columns:
        raise ValueError(
            f"Target '{target_column}' not found in training data."
        )

    if target_column not in test_df.columns:
        raise ValueError(
            f"Target '{target_column}' not found in test data."
        )

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Ensure test features have the same columns and ordering.
    if X_train.columns.tolist() != X_test.columns.tolist():
        raise ValueError(
            "Training and test feature columns do not match."
        )

    if X_train.isnull().any().any() or X_test.isnull().any().any():
        raise ValueError(
            "Missing feature values remain after preprocessing."
        )

    selected_task = detect_task(y_train) if task == "auto" else task

    if selected_task not in ("classification", "regression"):
        raise ValueError(
            "Task must be auto, classification, or regression."
        )

    return train_and_compare(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task=selected_task,
    )
