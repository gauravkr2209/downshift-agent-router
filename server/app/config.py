"""Configuration settings for Downshift."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment and .env file."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(default="sqlite:///downshift.db", alias="DATABASE_URL")
    nebius_api_key: str | None = Field(default=None, alias="NEBIUS_API_KEY")
    nebius_base_url: str | None = Field(default=None, alias="NEBIUS_BASE_URL")
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")

    ultra_model_id: str | None = Field(default=None, alias="ULTRA_MODEL_ID")
    super_model_id: str | None = Field(default=None, alias="SUPER_MODEL_ID")
    nano_model_id: str | None = Field(default=None, alias="NANO_MODEL_ID")

    prices_path: Path = Field(default=REPO_ROOT / "prices.yaml")

    nebius_request_timeout: float = Field(default=30.0, alias="NEBIUS_REQUEST_TIMEOUT")
    nebius_max_retries: int = Field(default=0, alias="NEBIUS_MAX_RETRIES")

    replay_retry_min_wait: float = Field(default=1.0, alias="REPLAY_RETRY_MIN_WAIT")
    replay_retry_max_wait: float = Field(default=30.0, alias="REPLAY_RETRY_MAX_WAIT")
    replay_retry_multiplier: float = Field(default=1.0, alias="REPLAY_RETRY_MULTIPLIER")
    replay_retry_attempts: int = Field(default=4, alias="REPLAY_RETRY_ATTEMPTS")


settings = Settings()
