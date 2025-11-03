from src.qa_agent import build_qa_agent

def main():
    try:
        file_path = "data/sample.pdf"  # Change to your document path
        queries = [
            "What is this document about?",
            "List key points mentioned.",
            "Who is the author or organization?"
        ]

        qa_agent = build_qa_agent(file_path)

        for query in queries:
            print(f"\n🔹 Query: {query}")
            answer = qa_agent.invoke({"query": query})
            print("Answer:", answer.get("result", "No result found."))

    except Exception as e:
        print(f"\n⚠️ Error occurred: {e}")

if __name__ == "__main__":
    main()
