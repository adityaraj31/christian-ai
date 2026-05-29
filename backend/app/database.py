import json
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

BIBLE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "bible_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_vector_store: Optional[FAISS] = None


def _load_all_verses() -> list[Document]:
    docs: list[Document] = []
    for json_path in sorted(BIBLE_DATA_DIR.glob("*.json")):
        with open(json_path, encoding="utf-8") as f:
            chapter_data = json.load(f)
        ref = chapter_data.get("reference", "")
        for verse in chapter_data.get("verses", []):
            page_content = verse["text"].strip()
            if not page_content:
                continue
            citation = f"{verse['book_name']} {verse['chapter']}:{verse['verse']}"
            metadata = {
                "citation": citation,
                "reference": ref,
                "book": verse["book_name"],
                "book_id": verse["book_id"],
                "chapter": verse["chapter"],
                "verse": verse["verse"],
                "translation_id": chapter_data.get("translation_id", "webbe"),
            }
            docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


def init_vector_store() -> None:
    global _vector_store
    docs = _load_all_verses()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    _vector_store = FAISS.from_documents(docs, embeddings)


def get_retriever(k: int = 4):
    if _vector_store is None:
        raise RuntimeError("Vector store not initialized. Call init_vector_store() first.")
    return _vector_store.as_retriever(search_kwargs={"k": k})
