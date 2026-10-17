
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from rag.csv_indexer import get_csv_vector_store


llm = ChatOllama(
    model="gemma3:1b",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI Data Analyst.

Answer the user's question using the retrieved CSV context.

Rules:
- Use simple, clear language.
- Use only facts supported by the retrieved context.
- Never invent records, values, or calculations.
- If the context is insufficient, say so clearly.
- Distinguish a few retrieved records from conclusions
  about the entire dataset.
- Do not claim that a value is the maximum, minimum,
  total, or average across the dataset unless the context
  supports that conclusion.
- Mention relevant CSV row or record numbers when available.
- Format the answer using Markdown.

Dataset context:
{context}

User question:
{question}

Write a helpful answer:
""")


def answer_csv_question(
    question: str,
    dataset_id: str,
    k: int = 5,
) -> dict:
    if not question or not question.strip():
        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    store = get_csv_vector_store(dataset_id)

    docs = store.similarity_search(
        question.strip(),
        k=k,
    )

    if not docs:
        return {
            "answer": (
                "I couldn't find relevant information "
                "in the indexed dataset."
            ),
            "sources": [],
        }

    context_parts = []

    for doc in docs:
        source = doc.metadata.get("source", "CSV")
        start = doc.metadata.get("row_start", "?")
        end = doc.metadata.get("row_end", "?")

        context_parts.append(
            f"Source: {source}\n"
            f"CSV lines: {start}-{end}\n"
            f"{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    response = llm.invoke(
        prompt.format_messages(
            context=context,
            question=question.strip(),
        )
    )

    sources = [
        {
            "filename": doc.metadata.get("source", "CSV"),
            "row_start": doc.metadata.get("row_start"),
            "row_end": doc.metadata.get("row_end"),
        }
        for doc in docs
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }
