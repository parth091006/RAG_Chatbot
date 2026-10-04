from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "advanced-rag")
    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    data_dir: str = os.getenv("DATA_DIR", "./data")
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "./data/processed/vector_store")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    @property
    def data_dir_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def vector_db_dir(self) -> Path:
        return Path(self.vector_db_path)


settings = Settings()
