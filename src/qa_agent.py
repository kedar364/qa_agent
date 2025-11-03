from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa import RetrievalQA
from langchain_community.llms import OpenAI


def build_qa_agent(file_path):
    """
    Builds a Retrieval-based QA Agent from a PDF document.
    """
    # Step 1: Load the document
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Step 2: Create embeddings
    embeddings = OpenAIEmbeddings()

    # Step 3: Create a vector store from the documents
    vector_store = FAISS.from_documents(documents, embeddings)

    # Step 4: Create a retriever and QA chain
    llm = OpenAI(temperature=0)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_agent = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return qa_agent
