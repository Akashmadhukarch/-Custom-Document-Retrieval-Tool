from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

def load_documents(path: str):
    loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    return loader.load()
