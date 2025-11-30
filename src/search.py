from src.vectorstore import FAISSVectorStore
from src.embedding import EmbeddingPipeline
from src.llm import get_llm
from langchain_core.messages import HumanMessage
from src.prompt import prompt_llm
import numpy as np

#retrive pipeline
class RAGSearch:
    def __init__(self, llm_model: str = "gpt-4o"):
        self.vectorstore = FAISSVectorStore()
        self.embedding_pipeline = EmbeddingPipeline()
        self.llm = get_llm(model=llm_model)
        print(f"[RAG] LLM ready: {llm_model}")

        # Auto-build if no index
        if self.vectorstore.index is None:
            print("[RAG] Building vector store from saved embeddings...")
            self.vectorstore.build_from_embeddings()

    def get_context(self, question: str, top_k: int = 5) -> str:
        query_emb = self.embedding_pipeline.model.encode(
            [question], normalize_embeddings=True
        ).astype("float32")
        results = self.vectorstore.search(query_emb, top_k)
        texts = [r["text"] for r in results]
        return "\n\n".join(texts)




    def _get_structured_context(self, question: str, top_k: int = 6):
            """Used by agents.py for logging — returns rich results"""
            query_emb = self.embedding_pipeline.model.encode(
                [question], normalize_embeddings=True
            ).astype("float32")
            
            return self.vectorstore.search(query_emb, top_k)  # Already returns list[dict]

    def index_file(self, file_path: str):
        """Index a new document and update the vector store"""
        from src.data_loader import load_all_documents

        # Load and chunk the document
        docs = load_all_documents([file_path])
        chunks = self.embedding_pipeline.splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No chunks generated from document")
        
        # Generate embeddings
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_pipeline.model.encode(
            texts, normalize_embeddings=True
        ).astype("float32")
        
        # Add to vector store
        self.vectorstore.add_embeddings(texts, embeddings, chunks)
        
        print(f"[RAG] Indexed {len(chunks)} chunks from {file_path}")

    def query(self, question: str, top_k: int = 5) -> str:
        context = self.get_context(question, top_k)
        if not context.strip():
            return "No relevant information found."

            # using llm

        prompt = prompt_llm.format(question=question, context=context)

        response = ""
        for chunk in self.llm.stream([HumanMessage(content=prompt)]):
            response += chunk.content
        return response.strip()

if __name__ == "__main__":
    rag = RAGSearch(llm_model="gpt-4.1")   # or "gpt-4o"
    print(rag.query("What is the main idea of 'Attention is All You Need'?", top_k=5))