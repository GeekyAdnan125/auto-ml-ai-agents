
import json
from pathlib import Path

from tools.data_tools import (
    load_dataset,
    profile_dataset,
    generate_recommendations,
    create_text_report,
)


BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"


def analyze_dataset(file_path: str) -> dict:
    """
    Coordinate dataset analysis and save the generated reports.
    """

    # Step 1: Load the input dataset.
    df = load_dataset(file_path)

    # Step 2: Calculate statistics and data-quality information.
    profile = profile_dataset(df)

    # Step 3: Generate rule-based recommendations.
    profile["recommendations"] = generate_recommendations(profile)

    # Step 4: Format the human-readable report.
    text_report = create_text_report(profile)

    # Step 5: Create the reports directory if necessary.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    text_path = REPORT_DIR / "analysis_report.txt"
    json_path = REPORT_DIR / "analysis_report.json"

    # Step 6: Save the text report.
    text_path.write_text(text_report, encoding="utf-8")

    # Step 7: Save the structured JSON report.
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            profile,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )

    return {
        "profile": profile,
        "text_report": text_report,
        "text_report_path": str(text_path),
        "json_report_path": str(json_path),
    }
