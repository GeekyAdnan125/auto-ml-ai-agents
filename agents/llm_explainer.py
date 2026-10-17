
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def explain_results(question: str, model_result: dict) -> str:
    """Explain AutoML results using a local Ollama model."""

    try:
        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gemma3:1b"),
            temperature=0,
        )

        messages = [
            SystemMessage(
                content=(
                    "You are an AI assistant for an AutoML application. "
                    "Explain machine learning results in simple English. "
                    "Use only the supplied training result. "
                    "Never invent metrics or dataset facts. "
                    "For regression, lower MAE and RMSE are generally "
                    "better, while higher R2 is generally better. "
                    "Explain which model performed best according to "
                    "the supplied selection metric. Mention limitations."
                )
            ),
            HumanMessage(
                content=(
                    f"User question: {question}\n\n"
                    f"Model training result:\n{model_result}"
                )
            ),
        ]

        response = llm.invoke(messages)
        return str(response.content)

    except Exception as exc:
        print(f"[Ollama warning] {exc}")
        return (
            "The AutoML pipeline may have completed, but the local "
            f"AI explanation could not be generated: {exc}\n\n"
            f"Model training result: {model_result}"
        )
