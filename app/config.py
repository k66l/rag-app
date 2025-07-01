import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    APP_NAME: str = "RAG Application"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Security settings
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # OpenAI settings (fallback)
    OPENAI_API_KEY: Optional[str] = None

    # Google AI Studio settings (primary)
    GOOGLE_API_KEY: Optional[str] = None
    GOOGLE_MODEL: str = "gemini-1.5-flash"  # Free model with generous quota

    # LLM Provider preference (google, openai, local)
    LLM_PROVIDER: str = "google"  # Default to Google AI Studio

    # Embedding settings
    USE_LOCAL_EMBEDDINGS: bool = True
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Document processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5

    TEMP_DOCS_DIR: str = "temp_docs"
    LLM_MODEL: str = "gpt-3.5-turbo"

    CHAIN_TYPE: str = "stuff"

    EMAIL: str = ""
    PASSWORD: str = ""
    JWT_SECRET: str = "your-secret-key-change-this-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24

    ADMIN_EMAIL: str = "admin@example.com"
    ADMIN_PASSWORD: str = "admin123"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
