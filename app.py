from src.qa_agent import build_qa_agent

def main():
    file_path = "data/sample.pdf"   # change to your file
    query = "What is this document about?"

    qa_agent = build_qa_agent(file_path)
    answer = qa_agent.invoke({"query": query})
    print("\nAnswer:\n", answer["result"])

if __name__ == "__main__":
    main()
