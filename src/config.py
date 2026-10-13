"""Project configuration using Pydantic settings."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    data_dir: Path = Field(default=Path("artifacts"), description="Path to raw/processed data")
    cache_dir: Path = Field(default=Path("data/cache"), description="Location for cached API responses")
    models_dir: Path = Field(default=Path("models"), description="Directory for persisted models")
    reports_dir: Path = Field(default=Path("reports"), description="Directory for generated reports")
    binance_base_url: str = Field(
        default="https://data-api.binance.vision/api",
        description="Base URL for Binance public data API",
    )
    deribit_base_url: str = Field(
        default="https://www.deribit.com/api/v2",
        description="Base URL for Deribit public API",
    )
    deribit_dvol_resolution: str = Field(
        default="1D",
        description="Default resolution for Deribit DVOL fetches",
    )
    deribit_dvol_index: str = Field(
        default="DVOL",
        description="Default Deribit volatility index name",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    model_config = SettingsConfigDict(env_prefix="HMM_", env_file=".env", env_file_encoding="utf-8")


def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()


settings = get_settings()
