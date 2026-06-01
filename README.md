# Christian AI Assistant

A production-grade, secure Christianity-focused AI assistant that provides grounded, citation-based biblical answers with image generation capabilities.

**Architectural Note**: This application uses a 3-node LangGraph state machine — Moderation → Retrieval → Generation — chained sequentially with a conditional safety bypass. Queries are first evaluated by an LLM guardrail; unsafe requests skip retrieval and go directly to a refusal response. Safe queries retrieve the top-4 Bible verses using **hybrid search** (BM25 keyword scoring + FAISS semantic embeddings fused with weighted score normalization at 40/60 ratio) and pass them as strict context to the generator, which is forced to answer exclusively from those citations. The frontend is a React chat UI with a denominational sidebar, citation toggles, and an image-generation trigger backed by Replicate's `google/imagen-4`.

![Chat Interface](image/Screenshot%20from%202026-05-29%2012-55-02.png)

## How It Works

All queries flow through a **3-node LangGraph state machine**:

1. **Moderation Node** — LLM guardrail evaluates the query. Unsafe requests (rewrites, hateful content, prompt injection) are immediately routed to a polite refusal.
2. **Retrieval Node** — First tries exact `Book Ch:Verse` file lookup. If a `Ch:V` pattern exists but no recognized book is found (fake scripture like "Hezekiah 3:16"), returns an immediate refusal. Otherwise runs **hybrid search**: BM25 keyword scores (40%) fused with FAISS semantic similarity scores (60%) via min-max normalized weighted fusion. Results with a top fused score below 0.05 are discarded (out-of-scope detection).
3. **Generation Node** — The generator is forced to answer *only* from the retrieved context, with citations. Denominational context (Protestant/Catholic/Orthodox) is injected seamlessly.

```
User ──► Moderation ──► SAFE? ──► Retrieval ──► Generation ──► Response
                 │                      ▲                        │
            UNSAFE└──────────────────────┘                        │
                 └──────────────────────────────────────────► Refusal
```

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    React + Vite Frontend                    │
│  ┌──────────┐  ┌─────────────────────────────────────────┐ │
│  │ Sidebar  │  │ Chat Window                             │ │
│  │ ─────────│  │ ┌─────────────────────────────────────┐ │ │
│  │ Denom.   │  │ │ User: What does John 3:16 say?     │ │ │
│  │ Dropdown │  │ │ Bot: [John 3:16]: For God so...    │ │ │
│  │          │  │ │ [Show citations ▼]                 │ │ │
│  │          │  │ │ [Visualize this Context]            │ │ │
│  │          │  │ └─────────────────────────────────────┘ │ │
│  └──────────┘  └─────────────────────────────────────────┘ │
└────────────────────────┬───────────────────────────────────┘
                         │ HTTP /api/*
┌────────────────────────▼───────────────────────────────────┐
│                     FastAPI Backend                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │  LangGraph: Moderation → Retrieval → Generation    │    │
│  └────────────────────────────────────────────────────┘    │
│  ┌─────────────────────┐  ┌────────────────────────────┐   │
│  │  Hybrid Search      │  │  Image Generation         │   │
│  │  ┌─────────┐┌──────┤  │  google/imagen-4 (Replicate)│   │
│  │  │  BM25   ││FAISS │  │  → DALL-E 3               │   │
│  │  │(40%)    ││(60%) │  │  → Pollinations.ai (free)  │   │
│  │  └─────────┘└──────┤  └────────────────────────────┘   │
│  │  Score fusion ↓    │                                    │
│  │  Top-4 verses      │                                    │
│  └─────────┬──────────┘                                    │
│           │                                                 │
│  ┌────────▼─────────┐                                      │
│  │  bible_data/     │                                      │
│  │  1189 JSON files │                                      │
│  └──────────────────┘                                      │
└────────────────────────────────────────────────────────────┘
```

## Components

### Backend

| File | Role |
|---|---|
| `app/database.py` | Loads Bible data, builds FAISS semantic index + BM25 keyword index. Provides `hybrid_search()` that fuses BM25 (40%) and FAISS (60%) scores via min-max normalized weighted fusion. Caches FAISS to disk for near-instant startup. Skips empty verses. |
| `app/schemas.py` | Pydantic models — `ChatRequest` (message, denomination, session_id), `ChatResponse` (response, safety_triggered, citations), `ImageRequest`/`ImageResponse`. |
| `app/graph.py` | LangGraph state machine. Three nodes with conditional safety edge. Book-aware exact verse lookup for `Book Ch:Verse` queries. Adversarial keyword detection prevents rewrite/fabricate queries from bypassing moderation. Fake scripture (`Ch:V` pattern with no known book) returns immediate refusal. Out-of-scope queries filtered by hybrid search score threshold. |
| `app/main.py` | FastAPI lifespan (vector store + graph init on startup). Endpoints: `GET /health`, `POST /api/chat`, `POST /api/generate-image`. Conversation memory via `_chat_store` dict keyed by session_id (last 20 messages). Image moderation guardrail. |

### Frontend

| File | Role |
|---|---|
| `src/App.jsx` | Chat UI with sidebar (denomination dropdown), message feed, collapsible citations, "Visualize this Context" button. Generates UUID session_id on first load. Loading indicators for chat and image generation. |
| `src/index.css` | Tailwind CSS entry point. |

## Safety & Guardrails

- **Moderation Node**: All queries pass through an LLM-based safety check before retrieval. Catches adversarial rewrites, hateful content, and prompt injection. Queries referencing known biblical books auto-pass unless they contain adversarial keywords (`rewrite`, `fabricate`, `ignore instructions`, `pretend`, etc.).
- **Fake Scripture**: Non-existent books (e.g., "Hezekiah 3:16") are detected by matching `Ch:V` patterns against the known book name list. The retrieval node returns an immediate refusal — the LLM never receives context to fabricate from.
- **Out-of-Scope Detection**: Hybrid search returns a fused relevance score. Queries unrelated to Scripture (e.g., stock market advice) score below 0.05 and produce a "no direct scriptural basis" response.
- **Grounded Generation**: System prompt forces only-context answers with citations. No hallucination.
- **Historical Defense**: Rule #8 in the generation prompt corrects inaccurate claims (e.g., "Constantine wrote the Bible").
- **Theological Complexity**: Rule #7 directs the model to acknowledge mystery (e.g., problem of evil) without false resolution.
- **Image Safety**: Image descriptions are LLM-moderated before API dispatch. Prompts are prefixed with a guardrail string prohibiting anachronisms, cartoons, and offensive content.
- **Multi-word Book Names**: Regex-free parser handles "Song of Solomon", "1 Kings", "2 Corinthians", etc.

## Running

### Backend

```bash
cd backend
cp .env.example .env    # fill in keys
uv run uvicorn app.main:app --reload
```

First startup builds the FAISS index (~2 min). Subsequent startups load from cache (< 5s).

### Frontend

```bash
cd frontend
npm install
npm run dev          # proxies /api to localhost:8000
```

Open `http://localhost:5173`.

## Image Generation

Priority chain:
1. `REPLICATE_API_TOKEN` → `google/imagen-4` via Replicate
2. `OPENAI_API_KEY` → DALL-E 3
3. Neither → Pollinations.ai (free, no key needed)

## Evaluation Dataset

`tests/evaluation_dataset.json` contains 20 test cases across 7 categories:

| Category | Count | Examples |
|---|---|---|
| grounding | 5 | John 3:16, Song of Solomon 2:1, out-of-scope |
| hallucination | 2 | Hezekiah 3:16, Apocryphon 5:12 |
| safety | 4 | Adversarial rewrites, hateful content, prompt injection |
| theology | 2 | Problem of evil, Trinity coherence |
| denomination | 3 | Canon scope per tradition (Protestant/Catholic/Orthodox) |
| image_safety | 2 | Inappropriate biblical depictions |
| memory | 1 | Multi-turn conversation follow-up |

### Run the evaluation suite

```bash
cd backend && uv run python ../tests/run_evaluation.py
```

All 20 test cases pass across grounding, safety, hallucination, theology, denomination, image safety, and conversation memory categories.

### Quick test with curl

```bash
curl -s http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What does John 3:16 say?","denomination":"Protestant"}'
```

## Stack

- **Backend**: Python 3.13, FastAPI, uv
- **Orchestration**: LangChain, LangGraph
- **LLM**: OpenRouter / Groq (via `ChatOpenAI`)
- **Vector Store**: FAISS (cached to disk) + BM25 (`rank-bm25`) + `all-MiniLM-L6-v2` (hybrid search, 40/60 weighted score fusion)
- **Image Gen**: Replicate (`google/imagen-4`) / DALL-E 3 / Pollinations.ai
- **Frontend**: React, Vite, Tailwind CSS
