from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

search_payh = "https://www.cas.cn/cm/202602/t20260205_5099813.shtml"

bs4_strainer = bs4.SoupStrainer()

loader = WebBaseLoader(
    web_path=search_payh,
    bs_kwargs={"parse_only": bs4_strainer},
)

docs = loader.load()

text_spiltter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100, add_start_index=True
)

all_splits = text_spiltter.split_documents(docs)

embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# doc_ids = vector_store.add_documents(documents=all_splits)

