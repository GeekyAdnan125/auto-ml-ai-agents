
from pathlib import Path
import sys

from graph.workflow import run_automl_workflow


BASE_DIR = Path(__file__).resolve().parent


def main():
    dataset_arg = (
        sys.argv[1] if len(sys.argv) > 1 else "data/sample.csv"
    )
    target_column = (
        sys.argv[2] if len(sys.argv) > 2 else "marks"
    )
    task = sys.argv[3].lower() if len(sys.argv) > 3 else "auto"
    question = (
        sys.argv[4]
        if len(sys.argv) > 4
        else "Explain the model comparison results and limitations."
    )

    dataset_path = Path(dataset_arg)

    if not dataset_path.is_absolute():
        dataset_path = BASE_DIR / dataset_path

    try:
        result = run_automl_workflow(
            dataset_path=str(dataset_path),
            target_column=target_column,
            task=task,
            user_question=question,
        )

        validation = result.get("validation_result", {})

        print("\n" + "=" * 60)
        print("AUTOML AGENT SYSTEM")
        print("=" * 60)

        if not validation.get("valid", False):
            print("Status: STOPPED")
            print("Reason:", result.get("error", "Validation failed"))
            sys.exit(1)

        model_result = result.get("model_result", {})

        print("Status: COMPLETED")
        print("Task:", model_result.get("task", task))
        print("Best model:", model_result.get("best_model", "See report"))

        print("\nLLM EXPLANATION")
        print("-" * 60)
        print(result.get("llm_answer", "No explanation returned."))

        print("\nReports:", BASE_DIR / "reports")
        print("Models:", BASE_DIR / "models")

    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        print(f"\nWorkflow failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
