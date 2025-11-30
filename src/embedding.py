# src/embedding.py
import os
import pickle
from pathlib import Path
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents

class EmbeddingPipeline:

    #chunking
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.model = SentenceTransformer(model_name)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.embed_dir = Path("data/embeddings")
        self.embed_dir.mkdir(exist_ok=True)

    #embeddings : for embedding we will use hugging face's sentence transformer model    

    def get_embed_path(self, file_path: str) -> Path:
        # example: data/docs/paper.pdf → data/embeddings/paper.pdf.pkl
        stem = Path(file_path).relative_to("data").with_suffix(".pkl")
        return self.embed_dir / stem

    def file_already_embedded(self, file_path: str) -> bool:
        return self.get_embed_path(file_path).exists()

    def save_embeddings(self, file_path: str, chunks: List, embeddings: np.ndarray):
        pkl_path = self.get_embed_path(file_path)
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pkl_path, "wb") as f:
            pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)
        print(f"[Saved] {file_path} → {pkl_path}")

    def run_on_new_files(self, data_folder: str = "data"):
        all_files = [str(p) for p in Path(data_folder).rglob("*") if p.is_file() and not str(p).startswith("data/embeddings")]
        new_files = 0

        for file_path in all_files:
            if self.file_already_embedded(file_path):
                print(f"[Skip] Already embedded: {file_path}")
                continue

            print(f"[Processing] {file_path}")
            docs = load_all_documents([file_path])  # load only this one file
            chunks = self.splitter.split_documents(docs)
            if not chunks:
                continue

            embeddings = self.model.encode(
                [c.page_content for c in chunks],
                show_progress_bar=False,
                normalize_embeddings=True
            ).astype('float32')

            self.save_embeddings(file_path, chunks, embeddings)
            new_files += 1

        print(f"\n[Done] Processed {new_files} new files. All embeddings are up to date!")

# Run this every time – it’s safe and fast
if __name__ == "__main__":
    pipeline = EmbeddingPipeline()
    pipeline.run_on_new_files()