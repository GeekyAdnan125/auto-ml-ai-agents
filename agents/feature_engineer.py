
from sklearn.model_selection import train_test_split

from tools.data_tools import load_dataset
from tools.feature_tools import (
    prepare_features,
    save_preprocessing_artifacts,
)


def engineer_features(
    file_path: str,
    target_column: str,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict:
    """
    Prepare a dataset for supervised machine learning.

    The target column is kept separate from input features.
    """

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    df = load_dataset(file_path)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    original_rows = len(df)

    # A supervised model cannot learn from rows without a target.
    df = df.dropna(subset=[target_column]).copy()
    dropped_target_rows = original_rows - len(df)

    if len(df) < 2:
        raise ValueError(
            "At least two rows with valid target values are required."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if X.shape[1] == 0:
        raise ValueError("No feature columns are available.")

    # Split before fitting any preprocessing steps.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )
    except ValueError as error:
        raise ValueError(
            f"Could not split the dataset: {error}"
        ) from error

    # Learn transformations from training features only.
    (
        X_train_prepared,
        X_test_prepared,
        preprocessor,
        all_missing_columns,
    ) = prepare_features(X_train, X_test)

    artifacts = save_preprocessing_artifacts(
        preprocessor=preprocessor,
        X_train_prepared=X_train_prepared,
        X_test_prepared=X_test_prepared,
        y_train=y_train,
        y_test=y_test,
        target_column=target_column,
    )

    return {
        "target_column": target_column,
        "original_rows": original_rows,
        "rows_without_target_removed": dropped_target_rows,
        "training_rows": len(X_train_prepared),
        "test_rows": len(X_test_prepared),
        "original_feature_count": X.shape[1],
        "prepared_feature_count": X_train_prepared.shape[1],
        "all_missing_columns_removed": all_missing_columns,
        "prepared_feature_names": X_train_prepared.columns.tolist(),
        "artifacts": artifacts,
    }
