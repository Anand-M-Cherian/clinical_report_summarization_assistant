from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    model: str
    log_level: str


def _load_settings() -> Settings:
    google_api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is required — no offline mode is supported"
        )
    return Settings(
        google_api_key=google_api_key,
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        log_level=os.environ.get("CLINICAL_ASSISTANT_LOG_LEVEL", "INFO"),
    )


settings = _load_settings()

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
