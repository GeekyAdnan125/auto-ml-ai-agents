
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"


def detect_task(y: pd.Series) -> str:
    """Infer task type from target dtype."""

    if (
        pd.api.types.is_object_dtype(y)
        or pd.api.types.is_bool_dtype(y)
        or isinstance(y.dtype, pd.CategoricalDtype)
    ):
        return "classification"

    return "regression"


def build_models(task: str) -> dict:
    """Return candidate models for the selected task."""

    if task == "classification":
        return {
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
            ),
        }

    if task == "regression":
        return {
            "linear_regression": LinearRegression(),
            "ridge": Ridge(),
            "random_forest": RandomForestRegressor(
                n_estimators=100,
                random_state=42,
            ),
        }

    raise ValueError(
        "task must be 'classification' or 'regression'."
    )


def evaluate_model(model, X_test, y_test, task: str) -> dict:
    """Evaluate a trained estimator on the held-out test set."""

    predictions = model.predict(X_test)

    if task == "classification":
        return {
            "accuracy": round(
                float(accuracy_score(y_test, predictions)), 4
            ),
            "f1_macro": round(
                float(
                    f1_score(
                        y_test,
                        predictions,
                        average="macro",
                        zero_division=0,
                    )
                ),
                4,
            ),
        }

    mse = mean_squared_error(y_test, predictions)

    return {
        "mae": round(
            float(mean_absolute_error(y_test, predictions)), 4
        ),
        "rmse": round(float(mse ** 0.5), 4),
        "r2": (
            round(float(r2_score(y_test, predictions)), 4)
            if len(y_test) >= 2
            else None
        ),
    }


def train_and_compare(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str,
) -> dict:
    """Train candidate models and select one by validation metric."""

    if len(X_train) < 2:
        raise ValueError("At least two training rows are required.")

    if X_train.empty or X_test.empty:
        raise ValueError("Training and test features must not be empty.")

    if y_train.nunique() < 2 and task == "classification":
        raise ValueError(
            "Classification requires at least two target classes "
            "in the training data."
        )

    label_encoder = None

    if task == "classification":
        # Encode string labels for model compatibility.
        label_encoder = LabelEncoder()
        y_train_fit = label_encoder.fit_transform(y_train.astype(str))
        y_test_fit = label_encoder.transform(y_test.astype(str))

    else:
        if not pd.api.types.is_numeric_dtype(y_train):
            raise ValueError(
                "Regression requires a numeric target. "
                "Choose classification for categorical targets."
            )

        y_train_fit = y_train.astype(float)
        y_test_fit = y_test.astype(float)

    models = build_models(task)
    results = {}
    successful_models = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train_fit)

            metrics = evaluate_model(
                model, X_test, y_test_fit, task
            )

            results[name] = {
                "status": "success",
                "metrics": metrics,
            }
            successful_models[name] = model

        except (ValueError, TypeError, ArithmeticError) as error:
            results[name] = {
                "status": "failed",
                "error": str(error),
            }

    if not successful_models:
        raise RuntimeError(
            "All candidate models failed. Check the dataset and target."
        )

    # Higher F1 is better for classification; higher R2 is better
    # for regression. If R2 is undefined for a tiny test set,
    # use lower MAE as the fallback.
    if task == "classification":
        best_name = max(
            successful_models,
            key=lambda name: results[name]["metrics"]["f1_macro"],
        )
        selection_metric = "f1_macro"

    else:
        models_with_r2 = [
            name for name in successful_models
            if results[name]["metrics"]["r2"] is not None
        ]

        if models_with_r2:
            best_name = max(
                models_with_r2,
                key=lambda name: results[name]["metrics"]["r2"],
            )
            selection_metric = "r2"
        else:
            best_name = min(
                successful_models,
                key=lambda name: results[name]["metrics"]["mae"],
            )
            selection_metric = "mae"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "best_model.joblib"
    report_path = REPORT_DIR / "model_comparison.json"

    # Save the fitted model and classification label encoder.
    joblib.dump(
        {
            "model": successful_models[best_name],
            "label_encoder": label_encoder,
            "task": task,
            "feature_names": X_train.columns.tolist(),
        },
        model_path,
    )

    report = {
        "task": task,
        "selection_metric": selection_metric,
        "best_model": best_name,
        "models": results,
        "model_path": str(model_path),
    }

    report_path.write_text(
        json.dumps(report, indent=4, allow_nan=False),
        encoding="utf-8",
    )

    return report
