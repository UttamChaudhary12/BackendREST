import hashlib
from datetime import datetime, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import HTTPException
from pydantic import ValidationError
from qdrant_client import models
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.database import Booking, Database, Document
from app.documents import chunk_text, extract_text
from app.providers import LLM, Memory, VectorStore
from app.schemas import BookingDetails, ChatRequest, ChatResponse, ChunkStrategy, IngestionResponse


class Services:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db = Database(settings.database_url)
        self.llm = LLM(settings)
        self.vectors = VectorStore(settings)
        self.memory = Memory(settings)

    async def ingest(
        self, data: bytes, filename: str, strategy: ChunkStrategy, size: int, overlap: int
    ) -> IngestionResponse:
        extension = "." + filename.rsplit(".", 1)[-1].lower()
        if extension not in {".pdf", ".txt"}:
            raise HTTPException(415, "Only PDF and UTF-8 TXT files are supported")
        if overlap >= size:
            raise HTTPException(422, "Overlap must be smaller than chunk_size")
        try:
            text = await run_in_threadpool(extract_text, data, extension)
        except Exception as exc:
            raise HTTPException(
                422, "Cannot extract text; use a readable PDF or UTF-8 TXT"
            ) from exc
        if not text.strip():
            raise HTTPException(422, "No text found; scanned PDFs require OCR before upload")
        if len(text) > self.settings.max_text_chars:
            raise HTTPException(413, "Extracted text exceeds the configured limit")
        chunks = chunk_text(text, strategy, size, overlap)
        document_id = str(uuid4())
        async with self.db.sessions() as db:
            document = Document(
                id=document_id,
                filename=filename,
                sha256=hashlib.sha256(data).hexdigest(),
                strategy=strategy.value,
                chunk_size=size,
                overlap=overlap,
                chunks=len(chunks),
                status="pending",
            )
            db.add(document)
            await db.commit()
            try:
                for offset in range(0, len(chunks), 64):
                    batch = chunks[offset : offset + 64]
                    embeddings = await self.llm.embed(batch)
                    if len(embeddings) != len(batch):
                        raise ValueError("Embedding provider returned an incomplete batch")
                    await self.vectors.client.upsert(
                        self.settings.collection,
                        wait=True,
                        points=[
                            models.PointStruct(
                                id=str(uuid4()),
                                vector=embedding,
                                payload={
                                    "document_id": document_id,
                                    "filename": filename,
                                    "chunk_index": offset + index,
                                    "text": chunk,
                                },
                            )
                            for index, (chunk, embedding) in enumerate(zip(batch, embeddings))
                        ],
                    )
                document.status = "ready"
                await db.commit()
            except Exception:
                await db.rollback()
                document = await db.get(Document, document_id)
                if document:
                    document.status = "failed"
                    await db.commit()
                # Failed/pending document IDs are excluded from all retrievals.
                raise
        return IngestionResponse(
            document_id=document_id, filename=filename, strategy=strategy, chunks=len(chunks)
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        session_id = str(request.session_id)
        # Nonblocking lease avoids interleaving turns across application workers.
        lock = self.memory.client.lock(f"lock:{session_id}", timeout=300, blocking=False)
        if not await lock.acquire():
            raise HTTPException(409, "Another turn is in progress for this session")
        try:
            return await self._chat(request)
        finally:
            await lock.release()

    async def _chat(self, request: ChatRequest) -> ChatResponse:
        session_id = str(request.session_id)
        state = await self.memory.read(session_id)
        zone = ZoneInfo(self.settings.booking_timezone)
        plan = await self.llm.plan(
            state["history"], request.message, state["draft"], datetime.now(zone).date().isoformat()
        )
        sources = []
        booking_id = None
        if plan.booking_intent:
            draft = plan.draft.model_dump()
            missing = [key for key, value in draft.items() if not value]
            if missing:
                answer = "To book your interview, please provide: " + ", ".join(missing) + "."
                state["awaiting_confirmation"] = False
            else:
                try:
                    details = BookingDetails.model_validate(draft)
                    starts_at = datetime.combine(details.date, details.time, tzinfo=zone)
                    if details.time.tzinfo is not None:
                        raise ValueError("Use a local time without a UTC offset")
                    if starts_at <= datetime.now(zone):
                        raise ValueError("Please choose a future date and time")
                except (ValidationError, ValueError) as exc:
                    answer = f"Please correct the booking details: {exc}"
                    state["awaiting_confirmation"] = False
                else:
                    if (
                        plan.confirmed
                        and state.get("awaiting_confirmation")
                        and draft == state["draft"]
                    ):
                        async with self.db.sessions() as db:
                            existing = await db.scalar(
                                select(Booking).where(Booking.session_id == session_id)
                            )
                            if existing:
                                booking_id = existing.id
                                answer = (
                                    f"Your interview is already booked. Reference: {booking_id}."
                                )
                            else:
                                booking = Booking(
                                    session_id=session_id,
                                    name=details.name,
                                    email=str(details.email),
                                    starts_at=starts_at.astimezone(timezone.utc),
                                    timezone=self.settings.booking_timezone,
                                )
                                db.add(booking)
                                try:
                                    await db.commit()
                                except IntegrityError:
                                    await db.rollback()
                                    answer = (
                                        "That interview slot is unavailable. Choose another time."
                                    )
                                else:
                                    booking_id = booking.id
                                    answer = (
                                        f"Interview booked for {details.date} at {details.time} "
                                        f"({self.settings.booking_timezone}). Reference: {booking_id}."
                                    )
                        state["awaiting_confirmation"] = False
                    else:
                        answer = (
                            f"Please confirm your interview: {details.name}, {details.email}, "
                            f"{details.date} at {details.time} ({self.settings.booking_timezone})."
                        )
                        state["awaiting_confirmation"] = True
            state["draft"] = draft
        else:
            async with self.db.sessions() as db:
                query = select(Document.id).where(Document.status == "ready")
                if request.document_ids:
                    query = query.where(Document.id.in_([str(id_) for id_ in request.document_ids]))
                ids = list((await db.scalars(query)).all())
            if ids:
                vector = (await self.llm.embed([plan.query or request.message]))[0]
                sources = await self.vectors.search(vector, ids)
            answer = await self.llm.answer(state["history"], request.message, sources)
        state["history"].extend(
            [{"role": "user", "content": request.message}, {"role": "assistant", "content": answer}]
        )
        await self.memory.save(session_id, state)
        return ChatResponse(
            session_id=request.session_id,
            answer=answer,
            sources=sources,
            booking_id=booking_id,
            booking_draft=plan.draft if plan.booking_intent else None,
        )
