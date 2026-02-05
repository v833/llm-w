from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="llama3")
file_path = "./关于阵容选择的一些思考.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search("商人商业强化")

print(results[0])
