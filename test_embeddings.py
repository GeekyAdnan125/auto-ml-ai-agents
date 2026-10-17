
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector = embeddings.embed_query(
    "Explain the purpose of an AutoML system."
)

print("Embedding test successful!")
print("Vector length:", len(vector))
print("First five values:", vector[:5])
