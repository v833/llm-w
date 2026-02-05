from typing import List
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import chain


embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)


@chain
def retrieval(query: str) -> List[Document]:
    return vector_store.similarity_search_with_score(query, k=1)


r = retrieval.invoke("商人商业强化")

print(r)
