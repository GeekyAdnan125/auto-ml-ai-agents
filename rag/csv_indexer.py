import pandas as pd
from pathlib import Path
import hashlib
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "chroma_db"

EMBED_MODEL = "nomic-embed-text"


def get_collection_name(dataset_id: str) -> str:
    # Chroma collection names must be valid and bounded.
    safe_id = hashlib.sha256(
        dataset_id.encode("utf-8")
    ).hexdigest()[:20]

    return f"csv_{safe_id}"


def get_csv_vector_store(dataset_id: str) -> Chroma:
    return Chroma(
        collection_name=get_collection_name(dataset_id),
        embedding_function=OllamaEmbeddings(
            model=EMBED_MODEL
        ),
        persist_directory=str(DB_DIR),
    )


def index_csv(
    df,
    dataset_id: str,
    filename: str,
) -> int:
    if df.empty:
        raise ValueError("The uploaded CSV has no rows.")

    if len(df.columns) == 0:
        raise ValueError("The uploaded CSV has no columns.")

    # Convert rows into readable text while retaining
    # column names and approximate CSV line numbers.
    row_documents = []
    rows_per_group = 10

    for start in range(0, len(df), rows_per_group):
        end = min(start + rows_per_group, len(df))
        lines = [
            f"Dataset: {filename}",
            f"CSV data rows: {start + 2}-{end + 1}",
            f"Columns: {', '.join(map(str, df.columns))}",
            "",
        ]

        for position in range(start, end):
            row = df.iloc[position]

            lines.append(
                f"Record at CSV line {position + 2}:"
            )

            for column, value in row.items():
                if str(value) == "nan":
                    value = "Missing"
                lines.append(f"{column}: {value}")

            lines.append("")

        row_documents.append(
            Document(
                page_content="\n".join(lines),
                metadata={
                    "dataset_id": dataset_id,
                    "source": filename,
                    "row_start": start + 2,
                    "row_end": end + 1,
                },
            )
        )

    # Split larger row groups if needed.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(row_documents)

    # Stable IDs make repeated indexing of the same
    # dataset update matching chunks instead of duplicating them.
    ids = []

    for index, chunk in enumerate(chunks):
        content_hash = hashlib.sha256(
            chunk.page_content.encode("utf-8")
        ).hexdigest()[:16]

        ids.append(
            f"{dataset_id}_{index}_{content_hash}"
        )

    store = get_csv_vector_store(dataset_id)
    store.add_documents(documents=chunks, ids=ids)

    return len(chunks)
