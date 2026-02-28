from langchain.chains import RetrievalQA
from app.llm.groq_llm import get_llm

def build_retrieval_chain(retriever):
    llm = get_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
