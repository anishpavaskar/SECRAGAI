from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

EMBEDDINGS = OpenAIEmbeddings()
VECTOR_DB_PATH = "data/vectordb"

def load_chain():
    vectorstore = FAISS.load_local(VECTOR_DB_PATH, EMBEDDINGS)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = OpenAI(temperature=0, model_name="gpt-4o-mini")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=True
    )
    return chain

rag_chain = load_chain()

def query_rag(question: str):
    result = rag_chain({"query": question})
    return {
        "answer": result["result"],
        "sources": [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in result["source_documents"]
        ]
    }
```
