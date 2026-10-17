
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"


def prepare_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Build and fit preprocessing on training features only.

    Numeric columns:
        Median imputation + standard scaling

    Categorical columns:
        Most-frequent imputation + one-hot encoding
    """

    # Remove columns that are entirely empty in training data.
    all_missing = X_train.columns[X_train.isna().all()].tolist()

    if all_missing:
        X_train = X_train.drop(columns=all_missing)
        X_test = X_test.drop(columns=all_missing)

    if X_train.shape[1] == 0:
        raise ValueError(
            "No usable feature columns remain after removing "
            "all-missing columns."
        )

    numeric_columns = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        exclude="number"
    ).columns.tolist()

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        transformers.append((
            "numeric",
            numeric_pipeline,
            numeric_columns,
        ))

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ])

        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    # Fit ONLY on the training features.
    X_train_array = preprocessor.fit_transform(X_train)

    # Transform test features without fitting on them.
    X_test_array = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out().tolist()

    X_train_prepared = pd.DataFrame(
        X_train_array,
        columns=feature_names,
        index=X_train.index,
    )

    X_test_prepared = pd.DataFrame(
        X_test_array,
        columns=feature_names,
        index=X_test.index,
    )

    return (
        X_train_prepared,
        X_test_prepared,
        preprocessor,
        all_missing,
    )


def save_preprocessing_artifacts(
    preprocessor,
    X_train_prepared: pd.DataFrame,
    X_test_prepared: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_column: str,
):
    """Save prepared datasets and the fitted preprocessing pipeline."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve the original target values and row alignment.
    train_output = X_train_prepared.copy()
    train_output[target_column] = y_train

    test_output = X_test_prepared.copy()
    test_output[target_column] = y_test

    train_path = REPORT_DIR / "prepared_train.csv"
    test_path = REPORT_DIR / "prepared_test.csv"
    pipeline_path = MODEL_DIR / "preprocessing_pipeline.joblib"

    train_output.to_csv(train_path, index=False)
    test_output.to_csv(test_path, index=False)

    joblib.dump(preprocessor, pipeline_path)

    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "pipeline_path": str(pipeline_path),
    }
