from typing import List, Union

from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Transcripto AI SaaS"
    APP_ENV: str = "development"
    APP_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # CORS
    CORS_ORIGINS: Union[str, List[AnyHttpUrl]] = [
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [origin.strip() for origin in v.split(",")]
        return v

    # API Keys
    OPENAI_API_KEY: str = ""
    HF_TOKEN: str = ""
    GROQ_API_KEY: str = ""

    # AI Providers
    ASR_PROVIDER: str = "faster_whisper"
    SUMMARY_PROVIDER: str = "openai"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # Upload Limits
    MAX_UPLOAD_SIZE_MB: int = 50

    # Settings
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # Ignore unknown variables like FLASK_ENV
    )


settings = Settings()