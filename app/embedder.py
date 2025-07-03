from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

EMBEDDINGS = OpenAIEmbeddings()
VECTOR_DB_PATH = "data/vectordb"

def build_vectorstore(chunks: list[str]):
    store = FAISS.from_texts(chunks, EMBEDDINGS)
    store.save_local(VECTOR_DB_PATH)
