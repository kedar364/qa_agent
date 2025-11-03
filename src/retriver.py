from langchain_community.llms import OpenAI
from langchain.chains.retrieval_qa import RetrievalQA

def create_retriever(vector_store):
    llm = OpenAI(temperature=0)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain
