# Christian AI Assistant

A production-grade, secure Christianity-focused AI assistant that provides grounded, citation-based biblical answers.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React + Vite)                  │
│  ┌──────────┐  ┌──────────────────────────────────────────┐ │
│  │ Sidebar  │  │         Chat Window                      │ │
│  │ ─────────│  │  ┌──────────────────────────────────┐   │ │
│  │ Denom.   │  │  │ User: What does John 3:16 say?   │   │ │
│  │ Dropdown │  │  │ ──────────────────────────────── │   │ │
│  │          │  │  │ Bot: [John 3:16]: For God so     │   │ │
│  │          │  │  │ loved the world...               │   │ │
│  │          │  │  │ [Show citations ▼]               │   │ │
│  │          │  │  │ [Visualize this Context]         │   │ │
│  │          │  │  └──────────────────────────────────┘   │ │
│  └──────────┘  └──────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP /api/*
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  Backend (FastAPI + Python)                  │
│                                                              │
│  ┌────────────────────────────────────────────────────┐      │
│  │  LangGraph State Machine                           │      │
│  │                                                     │      │
│  │  User Query ──► Moderation Node ──► SAFE? ──►      │      │
│  │                    │                     │          │      │
│  │               UNSAFE│               Retrieval Node  │      │
│  │                    │                     │          │      │
│  │                    ▼                     ▼          │      │
│  │              Generation Node ◄──────────────────────│      │
│  │                    │                                │      │
│  │                    ▼                                │      │
│  │               Response + Citations                  │      │
│  └────────────────────────────────────────────────────┘      │
│                                                              │
│  ┌──────────────────────────┐   ┌────────────────────────┐   │
│  │  FAISS Vector Store     │   │  Image Generation      │   │
│  │  (HuggingFace Embeddings)│   │  (DALL-E 3 / Stability)│   │
│  │  all-MiniLM-L6-v2       │   │  Safety-prefixed       │   │
│  └──────────┬───────────────┘   │  prompts               │   │
│             │                   └────────────────────────┘   │
│             │                                                │
│  ┌──────────▼───────────────┐                                │
│  │  bible_data/             │                                │
│  │  (1189 chapter JSONs)    │                                │
│  └──────────────────────────┘                                │
└──────────────────────────────────────────────────────────────┘
```

## Components

### Backend

| File | Role |
|---|---|
| `app/database.py` | Scans `bible_data/`, parses chapter JSONs into LangChain `Document` objects, builds in-memory FAISS index using HuggingFace `all-MiniLM-L6-v2` embeddings. Exposes `get_retriever(k=4)`. |
| `app/schemas.py` | Pydantic models: `ChatRequest` (message, denomination), `ChatResponse` (response, safety_triggered, citations), `ImageRequest`/`ImageResponse`. |
| `app/graph.py` | LangGraph state machine with `AgentState` typed dict. Three nodes: **Moderation Guardrail** (LLM eval → SAFE/UNSAFE), **Retrieval Engine** (FAISS semantic search), **Grounded Generator** (strict context-only prompt with denominational awareness). Conditional edge routes UNSAFE directly to polite refusal. |
| `app/main.py` | FastAPI lifespan (init vector store + compile graph on startup). Endpoints: `GET /health`, `POST /api/chat`, `POST /api/generate-image`. CORS middleware. |

### Frontend

| File | Role |
|---|---|
| `src/App.jsx` | Main chat UI: sidebar with Denominational Context Profile dropdown, message feed, citations toggle, "Visualize this Context" button. |
| `src/index.css` | Tailwind CSS entry point. |

## Safety & Guardrails

- **Moderation Node**: All queries pass through an LLM-based safety check before retrieval. Catches adversarial rewrites, hateful content, and prompt injection.
- **Fake Scripture**: If a user requests a non-existent book (e.g., "Hezekiah 3:16"), the vector store returns nothing, and the generator states "This text or book does not exist within historical scriptural data."
- **Grounded Generation**: The system prompt forces the model to answer *only* from the retrieved context block, with explicit citations. No hallucination.
- **Image Safety**: All DALL-E 3 prompts are prefixed with a protective guardrail string prohibiting anachronisms, cartoons, and offensive elements.

## Running

### Backend

```bash
cd backend
cp .env.example .env    # fill in Groq / OpenRouter / OpenAI keys
uv run uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev          # proxies /api to localhost:8000
```

## Stack

- **Backend**: Python 3.13, FastAPI, uv
- **Orchestration**: LangChain, LangGraph
- **LLM**: Groq API / OpenRouter (via `ChatOpenAI`)
- **Vector Store**: FAISS (in-memory) + `all-MiniLM-L6-v2`
- **Image Gen**: DALL-E 3
- **Frontend**: React, Vite, Tailwind CSS
