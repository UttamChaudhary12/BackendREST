import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from app.config import Settings
from app.schemas import ChatRequest, ChatResponse, ChunkStrategy, IngestionResponse
from app.services import Services

logger = logging.getLogger(__name__)
api_key_header = APIKeyHeader(name="X-API-Key")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    Path("data").mkdir(exist_ok=True)
    services = Services(settings)
    app.state.services = services
    try:
        await services.db.initialize()
        await services.memory.client.ping()
        await services.vectors.initialize()
        yield
    finally:
        await services.llm.client.close()
        await services.memory.client.aclose()
        await services.vectors.client.close()
        await services.db.engine.dispose()


app = FastAPI(title="Document RAG & Interview Booking", version="1.0.0", lifespan=lifespan)


def get_services(request: Request, key: Annotated[str, Depends(api_key_header)]) -> Services:
    services: Services = request.app.state.services
    if not secrets.compare_digest(key, services.settings.api_key.get_secret_value()):
        raise HTTPException(403, "Invalid API key")
    return services


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Request failed: %s", request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=503, content={"detail": "A backend dependency failed. Retry later."}
    )


@app.post("/api/v1/documents", response_model=IngestionResponse, status_code=201)
async def ingest_document(
    services: Annotated[Services, Depends(get_services)],
    file: Annotated[UploadFile, File()],
    strategy: Annotated[ChunkStrategy, Form()] = ChunkStrategy.paragraph,
    chunk_size: Annotated[int, Form(ge=200, le=4000)] = 1200,
    overlap: Annotated[int, Form(ge=0, le=1000)] = 200,
) -> IngestionResponse:
    try:
        data = await file.read(services.settings.max_upload_bytes + 1)
        if len(data) > services.settings.max_upload_bytes:
            raise HTTPException(413, "File is too large")
        if not data:
            raise HTTPException(422, "File is empty")
        filename = (file.filename or "upload").replace("\\", "/").rsplit("/", 1)[-1]
        if len(filename) > 255:
            raise HTTPException(422, "Filename is too long")
        return await services.ingest(data, filename, strategy, chunk_size, overlap)
    finally:
        await file.close()


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, services: Annotated[Services, Depends(get_services)]
) -> ChatResponse:
    if not request.message.strip():
        raise HTTPException(422, "Message cannot be blank")
    return await services.chat(request)
