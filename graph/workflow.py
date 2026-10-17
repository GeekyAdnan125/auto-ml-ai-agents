
from langgraph.graph import StateGraph, START, END

from state.ml_state import AutoMLState
from agents.validator import validate_dataset
from agents.data_analyst import analyze_dataset
from agents.feature_engineer import engineer_features
from agents.model_trainer import train_models
from agents.llm_explainer import explain_results


def validator_node(state: AutoMLState) -> dict:
    print("\n[LangGraph] Running Dataset Validator...")

    result = validate_dataset(
        dataset_path=state["dataset_path"],
        target_column=state["target_column"],
        task=state.get("task", "auto"),
    )

    if not result["valid"]:
        print(f"[Validation Error] {result['message']}")
        return {
            "validation_result": result,
            "error": result["message"],
            "current_step": "validation_failed",
        }

    print(f"[Validation] {result['message']}")

    return {
        "validation_result": result,
        "task": result["task"],
        "current_step": "validation_completed",
    }


def route_after_validation(state: AutoMLState) -> str:
    if state.get("error"):
        return "stop"
    return "continue"


def data_analyst_node(state: AutoMLState) -> dict:
    print("\n[LangGraph] Running Data Analyst Agent...")
    result = analyze_dataset(state["dataset_path"])
    return {
        "analysis_result": result,
        "current_step": "data_analysis_completed",
    }


def feature_engineer_node(state: AutoMLState) -> dict:
    print("\n[LangGraph] Running Feature Engineer Agent...")
    result = engineer_features(
        file_path=state["dataset_path"],
        target_column=state["target_column"],
    )
    return {
        "feature_result": result,
        "current_step": "feature_engineering_completed",
    }


def model_trainer_node(state: AutoMLState) -> dict:
    print("\n[LangGraph] Running Model Trainer Agent...")
    result = train_models(
        target_column=state["target_column"],
        task=state.get("task", "auto"),
    )
    return {
        "model_result": result,
        "current_step": "model_training_completed",
    }


def llm_explainer_node(state: AutoMLState) -> dict:
    print("\n[LangGraph] Running LLM Explainer...")
    answer = explain_results(
        question=state.get(
            "user_question",
            "Explain the model comparison results and limitations.",
        ),
        model_result=state.get("model_result", {}),
    )
    return {
        "llm_answer": answer,
        "current_step": "completed",
    }


def build_automl_graph():
    workflow = StateGraph(AutoMLState)

    workflow.add_node("validator", validator_node)
    workflow.add_node("data_analyst", data_analyst_node)
    workflow.add_node("feature_engineer", feature_engineer_node)
    workflow.add_node("model_trainer", model_trainer_node)
    workflow.add_node("llm_explainer", llm_explainer_node)

    workflow.add_edge(START, "validator")

    workflow.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "continue": "data_analyst",
            "stop": END,
        },
    )

    workflow.add_edge("data_analyst", "feature_engineer")
    workflow.add_edge("feature_engineer", "model_trainer")
    workflow.add_edge("model_trainer", "llm_explainer")
    workflow.add_edge("llm_explainer", END)

    return workflow.compile()


def run_automl_workflow(
    dataset_path: str,
    target_column: str,
    task: str = "auto",
    user_question: str = (
        "Explain the model comparison results and limitations."
    ),
) -> dict:
    app = build_automl_graph()

    initial_state: AutoMLState = {
        "dataset_path": dataset_path,
        "target_column": target_column,
        "task": task,
        "user_question": user_question,
        "current_step": "started",
    }

    return app.invoke(initial_state)
