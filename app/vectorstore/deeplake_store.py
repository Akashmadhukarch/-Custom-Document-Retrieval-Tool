from langchain_community.vectorstores import DeepLake
from app.config import ACTIVELOOP_TOKEN, DEEPLAKE_DATASET_PATH
from app.embeddings.embedding_model import get_embedding_model

def create_vector_store(documents=None):
    embedding = get_embedding_model()
    vector_store = DeepLake(
        dataset_path=DEEPLAKE_DATASET_PATH,
        embedding=embedding,
        token=ACTIVELOOP_TOKEN,
        overwrite=True if documents else False
    )
    if documents:
        vector_store.add_documents(documents)
    return vector_store
