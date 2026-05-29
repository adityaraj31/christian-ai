import json
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

BIBLE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "bible_data"
INDEX_DIR = Path(__file__).resolve().parent / "data" / "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 512

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


def _build_and_save(docs: list[Document]) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    texts = [d.page_content for d in docs]
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_embeddings = embeddings.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
        print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} verses...")

    embedding_matrix = np.array(all_embeddings, dtype=np.float32)
    store = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, embedding_matrix)),
        embedding=embeddings,
        metadatas=[d.metadata for d in docs],
    )
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))
    print(f"  FAISS index saved to {INDEX_DIR} with {len(docs)} verses.")
    return store


def _load_or_build() -> FAISS:
    index_file = INDEX_DIR / "index.faiss"
    if index_file.exists():
        print("  Loading cached FAISS index from disk...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
    print("  No cached index found. Building from Bible data...")
    docs = _load_all_verses()
    return _build_and_save(docs)


def init_vector_store() -> None:
    global _vector_store
    _vector_store = _load_or_build()


def get_retriever(k: int = 4):
    if _vector_store is None:
        raise RuntimeError("Vector store not initialized. Call init_vector_store() first.")
    return _vector_store.as_retriever(search_kwargs={"k": k})
