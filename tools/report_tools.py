
import json
from pathlib import Path

from langchain_core.tools import tool


BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"


def _read_json_report(filename: str) -> str:
    """Read a JSON report safely."""

    report_path = REPORT_DIR / filename

    if not report_path.exists():
        return (
            f"Report '{filename}' was not found. "
            "Run the AutoML pipeline first."
        )

    try:
        with report_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return json.dumps(data, indent=2, default=str)[:12000]

    except (OSError, json.JSONDecodeError) as error:
        return f"Unable to read report: {error}"


@tool
def get_dataset_analysis() -> str:
    """Retrieve dataset quality information, missing values,
    duplicates, column details, and recommendations."""
    return _read_json_report("analysis_report.json")


@tool
def get_model_comparison() -> str:
    """Retrieve machine learning model comparison metrics
    and the selected best model."""
    return _read_json_report("model_comparison.json")
