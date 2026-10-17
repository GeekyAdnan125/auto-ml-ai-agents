
from rag.vector_store import index_reports
from rag.rag_chain import answer_question

try:
    count = index_reports()
    print(f"Indexed {count} document chunks.")

    question = input("\nAsk about your AutoML reports: ")
    result = answer_question(question)

    print("\nAnswer:\n")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print("-", source)

except Exception as exc:
    print("RAG error:", exc)
