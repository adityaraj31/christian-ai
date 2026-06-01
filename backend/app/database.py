import json
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

BIBLE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "bible_data"
INDEX_DIR = Path(__file__).resolve().parent / "data" / "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 512
HYBRID_K = 4
HYBRID_CANDIDATES = 10

_vector_store: Optional[FAISS] = None
_bm25: Optional[BM25Okapi] = None
_docs: list[Document] = []


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


def _build_bm25_index(docs: list[Document]) -> BM25Okapi:
    tokenized = [d.page_content.lower().split() for d in docs]
    return BM25Okapi(tokenized)


def init_vector_store() -> None:
    global _vector_store, _bm25, _docs
    _docs = _load_all_verses()
    _vector_store = _load_or_build()
    _bm25 = _build_bm25_index(_docs)
    print(f"  BM25 index built with {len(_docs)} documents.")


def hybrid_search(query: str, k: int = HYBRID_K) -> tuple[list[Document], list[float]]:
    if _vector_store is None or _bm25 is None:
        raise RuntimeError("Vector store not initialized. Call init_vector_store() first.")

    candidates = max(k * 3, HYBRID_CANDIDATES)

    semantic_results = _vector_store.similarity_search_with_relevance_scores(
        query, k=candidates
    )

    tokenized_query = query.lower().split()
    bm25_raw = _bm25.get_scores(tokenized_query)
    bm25_indices = sorted(range(len(bm25_raw)), key=lambda i: -bm25_raw[i])[:candidates]

    idx_to_doc = {}
    sem_map = {}
    sem_scores = []
    for doc, score in semantic_results:
        idx = next((i for i, d in enumerate(_docs)
                    if d.page_content == doc.page_content
                    and d.metadata.get("citation") == doc.metadata.get("citation")), None)
        if idx is not None:
            idx_to_doc[idx] = doc
            sem_map[idx] = score
            sem_scores.append(score)

    bm25_candidate_scores = [bm25_raw[i] for i in bm25_indices]

    def min_max_norm(values):
        if not values:
            return {}
        mn, mx = min(values), max(values)
        if mx == mn:
            return {v: 0.5 for v in values}
        return {v: (v - mn) / (mx - mn) for v in values}

    sem_norm = min_max_norm(sem_scores)
    bm25_norm = min_max_norm(bm25_candidate_scores)

    fused = {}

    for idx in sem_map:
        fused[idx] = 0.6 * sem_norm.get(sem_map[idx], 0)

    for idx in bm25_indices:
        fused[idx] = fused.get(idx, 0) + 0.4 * bm25_norm.get(bm25_raw[idx], 0)

    top_indices = sorted(fused, key=lambda i: -fused[i])[:k]

    seen = set()
    results = []
    out_scores = []
    for idx in top_indices:
        doc = idx_to_doc.get(idx, _docs[idx]) if idx < len(_docs) else _docs[idx]
        key = doc.metadata.get("citation", "")
        if key not in seen:
            seen.add(key)
            results.append(doc)
            out_scores.append(fused[idx])
        if len(results) >= k:
            break

    if len(results) < k:
        for idx in bm25_indices[:k]:
            doc = _docs[idx] if idx < len(_docs) else idx_to_doc.get(idx, _docs[0])
            key = doc.metadata.get("citation", "")
            if key not in seen:
                seen.add(key)
                results.append(doc)
                out_scores.append(0.0)
            if len(results) >= k:
                break

    return results, out_scores
