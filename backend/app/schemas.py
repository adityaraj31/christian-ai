from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    denomination: str = Field(default="Protestant", pattern=r"^(Protestant|Catholic|Orthodox)$")
    session_id: str = Field(default="default")


class Citation(BaseModel):
    text: str
    reference: str


class ChatResponse(BaseModel):
    response: str
    safety_triggered: bool = False
    citations: list[Citation] = []


class ImageRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


class ImageResponse(BaseModel):
    image_url: str
