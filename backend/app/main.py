"""
Entry point for the Pulse RAG API.

Run locally (from the backend/ directory):
    uvicorn app.main:app --reload

Interactive docs:
    http://127.0.0.1:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()  # real environment variables take precedence over .env

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.router import _build_reasoning_agent, router  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm up the RAG index and LLM client so the first request is fast."""
    try:
        await run_in_threadpool(_build_reasoning_agent)
    except Exception as exc:  # the API still starts; endpoints return 503
        logger.error("Reasoning agent warm-up failed: %s", exc)
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Pulse RAG API",
    description=(
        "Turns lab report PDFs and images into structured data and "
        "patient-friendly, RAG-grounded insights. Informational only — "
        "not medical advice."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — only the origins listed in ALLOWED_ORIGINS may call the API
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health", tags=["Health"], summary="Service health check")
def health_check() -> dict:
    """Returns 200 OK if the service is running."""
    return {"status": "ok", "service": "pulse-rag-api", "env": settings.app_env}
