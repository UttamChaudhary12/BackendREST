from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: SecretStr
    api_key: SecretStr
    chat_model: str = "openai/gpt-oss-20b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    embedding_cache_dir: str = "./data/models"
    database_url: str = "sqlite+aiosqlite:///./data/app.db"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    collection: str = "documents_bge_small_v1"
    booking_timezone: str = "Asia/Kathmandu"
    memory_ttl: int = Field(default=86400, ge=60)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_text_chars: int = 500_000
