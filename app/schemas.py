from datetime import date, time
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class ChunkStrategy(str, Enum):
    fixed = "fixed"
    paragraph = "paragraph"


class IngestionResponse(BaseModel):
    document_id: UUID
    filename: str
    strategy: ChunkStrategy
    chunks: int


class ChatRequest(BaseModel):
    session_id: UUID
    message: str = Field(min_length=1, max_length=8000)
    document_ids: list[UUID] = Field(default_factory=list, max_length=50)


class BookingDraft(BaseModel):
    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None


class TurnPlan(BaseModel):
    query: str
    booking_intent: bool
    draft: BookingDraft
    confirmed: bool


class BookingDetails(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: EmailStr
    date: date
    time: time


class Source(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


class ChatResponse(BaseModel):
    session_id: UUID
    answer: str
    sources: list[Source]
    booking_id: UUID | None = None
    booking_draft: BookingDraft | None = None
