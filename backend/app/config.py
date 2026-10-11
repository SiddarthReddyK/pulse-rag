"""
Central configuration for the Pulse RAG backend.

All settings come from environment variables (optionally loaded from a
``.env`` file by ``app.main``) so the same code runs locally, in Docker,
and in CI without edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent / "data"

# Languages the UI offers. The value is the gTTS language code used for audio.
# Validating against this list also stops arbitrary text from being injected
# into the LLM prompt through the ``language`` form field.
SUPPORTED_LANGUAGES: dict[str, str] = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Telugu": "te",
    "Marathi": "mr",
    "Tamil": "ta",
    "Urdu": "ur",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    # gTTS has no Odia voice, so audio falls back to Hindi.
    "Odia": "hi",
}


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    app_env: str
    allowed_origins: list[str]
    max_file_size_mb: int
    llm_model: str
    vision_model: str
    embedding_provider: str          # "fastembed" | "hashing"
    embedding_model: str
    knowledge_file: Path
    rag_top_k: int
    rag_max_chunks: int

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def groq_api_key(self) -> str:
        # Read lazily so tests and tooling can import the app without a key.
        return os.getenv("GROQ_API_KEY", "").strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        app_env=os.getenv("APP_ENV", "development"),
        allowed_origins=_split_csv(
            os.getenv(
                "ALLOWED_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173",
            )
        ),
        max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "10")),
        llm_model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        vision_model=os.getenv(
            "VISION_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct"
        ),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "fastembed").lower(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        knowledge_file=Path(
            os.getenv("KNOWLEDGE_FILE", str(_DATA_DIR / "medical_knowledge.txt"))
        ),
        rag_top_k=int(os.getenv("RAG_TOP_K", "5")),
        rag_max_chunks=int(os.getenv("RAG_MAX_CHUNKS", "8")),
    )
