from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

def load_all_documents(data_input: str | List[str]) -> List[Any]:
    """
    Accepts either:
    - a folder path (str) → loads all files inside
    - a list of file paths → loads only those files
    """
    documents = []

    # If it's a list of files (like ["data/pdfs/paper.pdf"])
    if isinstance(data_input, (list, tuple)):
        file_paths = [Path(p) for p in data_input]
    else:
        # It's a folder → find all files recursively
        data_path = Path(data_input).resolve()
        file_paths = [p for p in data_path.rglob("*") if p.is_file()]

    print(f"[INFO] Loading {len(file_paths)} file(s)...")

    for file_path in file_paths:
        file_path = file_path.resolve()
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() in [".txt", ".md"]:
                loader = TextLoader(str(file_path), encoding="utf-8")
            elif file_path.suffix.lower() == ".csv":
                loader = CSVLoader(str(file_path))
            elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                loader = UnstructuredExcelLoader(str(file_path))
            elif file_path.suffix.lower() == ".docx":
                loader = Docx2txtLoader(str(file_path))
            elif file_path.suffix.lower() == ".json":
                loader = JSONLoader(str(file_path), jq_schema=".[]", text_content=False)
            else:
                print(f"[Skip] Unsupported file type: {file_path}")
                continue

            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = str(file_path)  # important for tracking
            documents.extend(docs)
            print(f"[Loaded] {file_path.name} → {len(docs)} pages/rows")

        except Exception as e:
            print(f"[ERROR] Failed {file_path}: {e}")

    print(f"[Success] Total loaded: {len(documents)} document pages")
    return documents

    
# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)