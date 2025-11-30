# src/app.py  ← Best version
from src.search import RAGSearch
import textwrap

if __name__ == "__main__":
    print("\n[RAG Demo] Starting RAG system...\n")
    
    # Initialize once — handles everything (FAISS + LLM)
    rag = RAGSearch(llm_model="gpt-4.1")   # or "gpt-4o"

    print("\n[RAG Demo] Ready! Type your question (or 'quit' to exit)\n")
    print("—" * 80)

    while True:
        try:
            query = input("\nQuestion: ").strip()
            
            if query.lower() in ["quit", "exit", "bye", "q"]:
                print("\nGoodbye! Have a great day")
                break
            if not query:
                continue

            print("\nAnswer:")
            answer = rag.query(query, top_k=6)
            
            # Beautiful wrapped output
            wrapped = textwrap.fill(answer, width=80, replace_whitespace=False)
            print(wrapped)
            print("\n" + "—" * 80)

        except KeyboardInterrupt:
            print("\n\nStopped by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")