
import json
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from rag.vector_store import get_vector_store

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"

llm = ChatOllama(
    model="gemma3:1b",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template("""
You are an assistant for an AutoML application.

Answer in simple, clear language using the supplied context
and exact model comparison data.

Rules:
- Never invent metric values.
- Do not confuse MSE with RMSE.
- For MAE and RMSE, lower is better.
- For R2, higher is generally better.
- Use the selection metric specified in the data, if provided.
- If the winning model is not explicitly identified and the
  selection rule is unknown, say that the selection rule is unknown.
- Distinguish verified facts from suggestions.

Retrieved context:
{context}

Exact model comparison data:
{metrics}

User question:
{question}
""")


def load_model_comparison() -> str:
    """Read the exact model comparison report."""

    report_path = REPORTS_DIR / "model_comparison.json"

    if not report_path.exists():
        return "Model comparison report is unavailable."

    try:
        data = json.loads(
            report_path.read_text(encoding="utf-8")
        )
        return json.dumps(data, indent=2, ensure_ascii=False)

    except (json.JSONDecodeError, OSError) as exc:
        return f"Could not read model comparison report: {exc}"


def answer_question(question: str) -> dict:
    """Retrieve relevant documents and generate a grounded answer."""

    if not question.strip():
        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    vector_store = get_vector_store()

    docs = vector_store.similarity_search(
        question,
        k=4,
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    ) or "No relevant report passages were retrieved."

    metrics = load_model_comparison()

    response = llm.invoke(
        prompt.format_messages(
            context=context,
            metrics=metrics,
            question=question,
        )
    )

    sources = list(
        dict.fromkeys(
            doc.metadata.get("source", "Unknown")
            for doc in docs
        )
    )

    if "model_comparison.json" not in sources:
        if "unavailable" not in metrics.lower():
            sources.append("model_comparison.json")

    return {
        "answer": response.content,
        "sources": sources,
    }
