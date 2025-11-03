from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store
