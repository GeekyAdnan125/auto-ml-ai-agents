from typing import Any, TypedDict


class AutoMLState(TypedDict, total=False):
    dataset_path: str
    target_column: str
    task: str
    user_question: str

    validation_result: dict[str, Any]
    analysis_result: dict[str, Any]
    feature_result: dict[str, Any]
    model_result: dict[str, Any]
    llm_answer: str

    error: str
    current_step: str
