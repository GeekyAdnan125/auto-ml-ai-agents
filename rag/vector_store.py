
from pathlib import Path
import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
DB_DIR = BASE_DIR / "chroma_db"

EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "automl_knowledge"


def get_embeddings():
    """Create the local embedding model."""
    return OllamaEmbeddings(model=EMBED_MODEL)


def get_vector_store():
    """Open the persistent Chroma collection."""
    DB_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(DB_DIR),
    )


def index_reports() -> int:
    """Index reports without deleting the collection."""

    if not REPORTS_DIR.exists():
        raise FileNotFoundError(
            f"Reports folder not found: {REPORTS_DIR}"
        )

    documents = []

    for path in REPORTS_DIR.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in {".txt", ".json"}:
            continue

        if path.stat().st_size == 0:
            continue

        content = path.read_text(encoding="utf-8")

        documents.append(
            Document(
                page_content=content,
                metadata={"source": path.name},
            )
        )

    if not documents:
        raise ValueError(
            "No reports found. Run the AutoML pipeline first."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )

    chunks = splitter.split_documents(documents)

    # Stable IDs help update existing chunks on repeated indexing.
    ids = []

    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        raw_id = f"{source}\n{chunk.page_content}"
        chunk_id = hashlib.sha256(
            raw_id.encode("utf-8")
        ).hexdigest()

        ids.append(chunk_id)

    store = get_vector_store()
    store.add_documents(documents=chunks, ids=ids)

    return len(chunks)
