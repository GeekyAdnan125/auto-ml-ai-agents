
from pathlib import Path

import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load and validate a CSV dataset."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported.")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    if len(df.columns) == 0:
        raise ValueError("The dataset has no columns.")

    if len(df.columns) != len(set(df.columns)):
        raise ValueError("Duplicate column names are not supported.")

    return df


def profile_dataset(df: pd.DataFrame) -> dict:
    """Calculate dataset-level and column-level statistics."""

    rows, columns = df.shape
    missing = df.isnull().sum()
    duplicate_rows = int(df.duplicated().sum())

    column_details = {}

    for column in df.columns:
        series = df[column]
        non_missing = series.dropna()

        details = {
            "data_type": str(series.dtype),
            "missing_values": int(series.isnull().sum()),
            "missing_percentage": round(
                float(series.isnull().mean() * 100), 2
            ),
            "unique_values": int(series.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series):
            numeric = series.dropna()

            details["statistics"] = {
                "mean": (
                    round(float(numeric.mean()), 3)
                    if not numeric.empty else None
                ),
                "median": (
                    round(float(numeric.median()), 3)
                    if not numeric.empty else None
                ),
                "minimum": (
                    round(float(numeric.min()), 3)
                    if not numeric.empty else None
                ),
                "maximum": (
                    round(float(numeric.max()), 3)
                    if not numeric.empty else None
                ),
                "standard_deviation": (
                    round(float(numeric.std()), 3)
                    if len(numeric) > 1 else None
                ),
            }

        else:
            top_values = non_missing.value_counts().head(5)

            details["top_values"] = [
                {
                    "value": str(value),
                    "count": int(count),
                }
                for value, count in top_values.items()
            ]

        column_details[str(column)] = details

    return {
        "rows": int(rows),
        "columns": int(columns),
        "duplicate_rows": duplicate_rows,
        "missing_cells": int(missing.sum()),
        "missing_percentage": round(
            float(missing.sum() / (rows * columns) * 100), 2
        ),
        "numeric_columns": [
            str(c) for c in df.select_dtypes(include="number").columns
        ],
        "categorical_columns": [
            str(c) for c in df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns
        ],
        "column_details": column_details,
    }


def generate_recommendations(profile: dict) -> list:
    """Generate simple, rule-based data-quality recommendations."""

    recommendations = []

    if profile["missing_cells"] > 0:
        recommendations.append(
            "Handle missing values before training a model. "
            "Choose an appropriate strategy for each column."
        )

    if profile["duplicate_rows"] > 0:
        recommendations.append(
            "Review duplicate rows and remove them if they "
            "represent repeated records rather than valid observations."
        )

    for column, details in profile["column_details"].items():
        if details["missing_percentage"] >= 30:
            recommendations.append(
                f"Column '{column}' has "
                f"{details['missing_percentage']}% missing values. "
                "Review whether to impute or exclude it."
            )

        if (
            profile["rows"] > 0
            and details["unique_values"] == 1
        ):
            recommendations.append(
                f"Column '{column}' has only one unique non-missing "
                "value. Consider excluding it from model training."
            )

    if not recommendations:
        recommendations.append(
            "No basic data-quality issues were detected by these checks. "
            "Perform further validation before model training."
        )

    return recommendations


def create_text_report(profile: dict) -> str:
    """Convert analysis results into a readable report."""

    lines = [
        "=" * 60,
        "AI AUTOML SYSTEM - DATA ANALYSIS REPORT",
        "=" * 60,
        f"Rows: {profile['rows']}",
        f"Columns: {profile['columns']}",
        f"Duplicate rows: {profile['duplicate_rows']}",
        f"Missing cells: {profile['missing_cells']}",
        f"Missing data: {profile['missing_percentage']}%",
        "",
        "COLUMN ANALYSIS",
        "-" * 60,
    ]

    for column, details in profile["column_details"].items():
        lines.extend([
            f"\nColumn: {column}",
            f"  Type: {details['data_type']}",
            f"  Missing: {details['missing_values']}",
            f"  Missing percentage: {details['missing_percentage']}%",
            f"  Unique values: {details['unique_values']}",
        ])

        if "statistics" in details:
            lines.append("  Numeric statistics:")

            for key, value in details["statistics"].items():
                lines.append(f"    {key}: {value}")

        if "top_values" in details:
            lines.append("  Most frequent values:")

            for item in details["top_values"]:
                lines.append(
                    f"    {item['value']}: {item['count']}"
                )

    lines.extend([
        "",
        "RECOMMENDATIONS",
        "-" * 60,
    ])

    for index, recommendation in enumerate(
        profile["recommendations"], start=1
    ):
        lines.append(f"{index}. {recommendation}")

    lines.append("=" * 60)

    return "\n".join(lines)
