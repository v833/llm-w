from typing import List
import chromadb


def list_collections(db_path: str) -> List[chromadb.Collection]:
    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    return collections


def delete_collection(db_path: str, collection_name: str):
    try:
        client = chromadb.PersistentClient(path=db_path)
        client.delete_collection(collection_name)
    except Exception as e:
        print(f"删除集合 {collection_name} 失败: {e}")


# collections = list_collections("./chroma_langchain_db")

# for collection in collections:
#     print(collection.name, collection.count())
