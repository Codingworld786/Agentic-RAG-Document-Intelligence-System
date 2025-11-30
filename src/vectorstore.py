# src/vectorstore.py
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import os

class FAISSVectorStore:
    def __init__(self, persist_dir: str = "faiss_store"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        self.index_path = self.persist_dir / "faiss.index"
        self.metadata_path = self.persist_dir / "metadata.pkl"

        if self.index_path.exists() and self.metadata_path.exists():
            print("[VectorStore] Loading existing FAISS index...")
            self._load()
        else:
            print("[VectorStore] No index found. Will build when you call .build()")
            self.index = None
            self.metadata = []

    def _load(self):
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[VectorStore] Loaded {self.index.ntotal} vectors")

    def build_from_embeddings(self, embed_dir: str = "data/embeddings"):
        from src.embedding import EmbeddingPipeline
        pipeline = EmbeddingPipeline()

        all_embeddings = []
        all_metadatas = []
        embed_dir_path = Path(embed_dir)

        print(f"[VectorStore] Building from {embed_dir_path}")

        for pkl_file in embed_dir_path.rglob("*.pkl"):
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
                embeddings = data["embeddings"].astype("float32")
                chunks = data["chunks"]

                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings) #euclidean distance

                all_embeddings.append(embeddings)
                for chunk in chunks:
                    all_metadatas.append({
                        "text": chunk.page_content,
                        "source": chunk.metadata.get("source", str(pkl_file.stem))
                    })

        if not all_embeddings:
            raise ValueError("No embeddings found!")

        embeddings_matrix = np.vstack(all_embeddings)
        dim = embeddings_matrix.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner Product = cosine
        self.index.add(embeddings_matrix)
        self.metadata = all_metadatas

        # Save
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[VectorStore] Built and saved index with {len(self.metadata)} chunks")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.index is None:
            raise ValueError("Index not built or loaded!")
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                meta = self.metadata[idx]
                results.append({"text": meta["text"], "source": meta["source"], "score": float(score)})
        return results


    def add_embeddings(self, texts: List[str], embeddings: np.ndarray, chunks: List):
        """Add new embeddings to existing index"""
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create new metadata
        new_metadata = []
        for chunk in chunks:
            new_metadata.append({
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "uploaded_file")
            })
        
        # Add to index
        if self.index is None:
            # Create new index if none exists
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.metadata = []
        
        self.index.add(embeddings)
        self.metadata.extend(new_metadata)
        
        # Save updated index
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        
        print(f"[VectorStore] Added {len(new_metadata)} new chunks. Total: {self.index.ntotal}")
