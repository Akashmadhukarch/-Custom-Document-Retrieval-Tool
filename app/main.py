from app.ingestion.load_documents import load_documents
from app.ingestion.text_splitter import split_documents
from app.vectorstore.deeplake_store import create_vector_store
from app.retriever.retriever import get_retriever
from app.chains.retrieval_chain import build_retrieval_chain

def ingest_documents():
    docs = load_documents("data/sample_docs")
    split_docs = split_documents(docs)
    return create_vector_store(split_docs)

def main():
    print("Ingesting documents...")
    vector_store = ingest_documents()
    retriever = get_retriever(vector_store)
    qa_chain = build_retrieval_chain(retriever)

    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        result = qa_chain({"query": query})
        print("Answer:", result["result"])

if __name__ == "__main__":
    main()
