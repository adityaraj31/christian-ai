import os
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph import StateGraph

from app.database import init_vector_store
from app.graph import build_graph
from app.schemas import ChatRequest, ChatResponse, Citation, ImageRequest, ImageResponse

load_dotenv()

_graph: StateGraph | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    print("Initializing vector store from Bible data...")
    init_vector_store()
    print("Building LangGraph state machine...")
    _graph = build_graph()
    print("Application ready.")
    yield
    print("Shutting down.")


app = FastAPI(title="Christian AI Assistant", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if _graph is None:
        raise HTTPException(status_code=503, detail="Service not ready yet.")

    initial_state = {
        "query": req.message,
        "denomination": req.denomination,
        "safety_check": "PENDING",
        "retrieved_context": "",
        "citations": [],
        "response": "",
    }

    result = await _graph.ainvoke(initial_state)

    return ChatResponse(
        response=result["response"],
        safety_triggered=result["safety_check"] == "UNSAFE",
        citations=[Citation(**c) for c in result.get("citations", [])],
    )


IMAGE_SAFETY_PREFIX = (
    "A historically accurate, highly respectful, cinematic painting "
    "depicting the historical biblical scene of: "
)
IMAGE_SAFETY_SUFFIX = (
    "Strictly avoid cartoon styles, modern anachronisms, "
    "or offensive/heretical/toxic elements."
)


@app.post("/api/generate-image", response_model=ImageResponse)
async def generate_image(req: ImageRequest):
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="Image generation API key not configured.")

    safe_prompt = f"{IMAGE_SAFETY_PREFIX}{req.text}. {IMAGE_SAFETY_SUFFIX}"

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "dall-e-3",
                "prompt": safe_prompt,
                "n": 1,
                "size": "1024x1024",
            },
        )
        data = resp.json()
        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail=data.get("error", {}).get("message", "Image generation failed"),
            )

    image_url = data["data"][0]["url"]
    return ImageResponse(image_url=image_url)
