
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemma3:1b",
    temperature=0,
)

response = llm.invoke(
    "Explain what an R2 score means in machine learning "
    "using simple language and one example."
)

print(response.content)
