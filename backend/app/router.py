"""
API router for Pulse RAG.

POST /api/v1/upload-report pipeline:
  1. Validate the upload (type, size, language)
  2. Extract raw text            -> pdf_extractor / image_extractor
  3. Extract structured values   -> llm_service (Groq)
  4. Classify High / Low / Normal -> lab_report_analyzer
  5. Generate RAG-grounded insights -> reasoning_layer (Groq)
  6. Return the combined AnalysisResponse

POST /api/v1/analyze-trends runs steps 1-4 on two reports and compares them.
The (slower) insight generation in step 5 is skipped there on purpose.
"""

import io
import logging
import os
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from gtts import gTTS

from app import schemas
from app.config import SUPPORTED_LANGUAGES, get_settings
from app.graph_generator import generate_trend_graph_base64
from app.image_extractor import extract_text_from_image
from app.lab_report_analyzer import analyze_trends, classify_report_statuses
from app.llm_service import analyze_report_text
from app.pdf_extractor import extract_text_from_pdf
from app.reasoning_layer import LabReportReasoningAgent
from app.schemas import AnalysisResponse, TrendAnalysisResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Lab Report"])

IMAGE_MIME_BY_EXT = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tiff": "image/tiff", ".tif": "image/tiff",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}
IMAGE_CONTENT_TYPES = set(IMAGE_MIME_BY_EXT.values()) | {"image/jpg"}
PDF_MAGIC = b"%PDF"


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _build_reasoning_agent() -> LabReportReasoningAgent:
    return LabReportReasoningAgent()


def get_reasoning_agent() -> LabReportReasoningAgent:
    """Create the reasoning agent on first use (tests override this)."""
    try:
        return _build_reasoning_agent()
    except RuntimeError as exc:
        logger.error("Reasoning agent unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI reasoning service unavailable. Check GROQ_API_KEY and server logs.",
        ) from exc


def validate_language(language: str = Form("English", description="Output language")) -> str:
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language '{language}'.",
        )
    return language


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extension(file: UploadFile) -> str:
    return os.path.splitext((file.filename or "").lower())[1]


def _is_image(file: UploadFile) -> bool:
    return file.content_type in IMAGE_CONTENT_TYPES or _extension(file) in IMAGE_MIME_BY_EXT


def _image_mime(file: UploadFile) -> str:
    if file.content_type in IMAGE_CONTENT_TYPES:
        return "image/jpeg" if file.content_type == "image/jpg" else file.content_type
    return IMAGE_MIME_BY_EXT.get(_extension(file), "image/jpeg")


async def _read_upload(file: UploadFile) -> tuple[bytes, bool, str]:
    """Validate an upload and return (bytes, is_image, image_mime)."""
    name = file.filename or "file"
    is_image = _is_image(file)
    is_pdf_hint = file.content_type == "application/pdf" or _extension(file) == ".pdf"

    if not is_image and not is_pdf_hint:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF and image files (JPG, PNG, TIFF, BMP, WEBP) are accepted.",
        )

    limit = get_settings().max_file_size_bytes
    data = await file.read(limit + 1)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"File '{name}' is empty.")
    if len(data) > limit:
        raise HTTPException(
            413,
            detail=f"File '{name}' exceeds the {get_settings().max_file_size_mb} MB limit.",
        )
    if not is_image and not data.startswith(PDF_MAGIC):
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File '{name}' is not a valid PDF.",
        )
    return data, is_image, _image_mime(file) if is_image else ""


def _extract_and_classify(data: bytes, is_image: bool, mime: str) -> dict[str, Any]:
    """Blocking: text extraction -> LLM parameter extraction -> classification."""
    try:
        text = extract_text_from_image(data, mime_type=mime) if is_image else extract_text_from_pdf(data)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Text extraction failed: %s", exc)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Text extraction failed.")

    try:
        lab_report = analyze_report_text(text)
    except ValueError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=f"LLM extraction failed: {exc}")
    except RuntimeError as exc:
        logger.error("LLM service unavailable: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="LLM service unavailable.")

    return classify_report_statuses(lab_report.model_dump())


def _full_pipeline(
    data: bytes, is_image: bool, mime: str, language: str, agent: LabReportReasoningAgent
) -> dict[str, Any]:
    """Blocking: extraction + classification + RAG insights."""
    report = _extract_and_classify(data, is_image, mime)
    try:
        insights = agent.analyze(report, language=language)
    except Exception as exc:
        logger.exception("Reasoning layer failed.")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, detail="AI reasoning failed. Please try again."
        ) from exc
    return {**report, **insights}


# ---------------------------------------------------------------------------
# POST /api/v1/upload-report — single report
# ---------------------------------------------------------------------------

@router.post(
    "/upload-report",
    response_model=AnalysisResponse,
    summary="Upload a lab report and get full AI analysis",
    description=(
        "Accepts a PDF or image lab report. Extracts and classifies the lab "
        "parameters, then generates a patient-friendly summary, preventive "
        "guidance, and 3 questions to ask a doctor."
    ),
)
async def upload_report(
    file: UploadFile = File(..., description="Lab report — PDF or image"),
    language: str = Depends(validate_language),
    agent: LabReportReasoningAgent = Depends(get_reasoning_agent),
) -> AnalysisResponse:
    data, is_image, mime = await _read_upload(file)
    result = await run_in_threadpool(_full_pipeline, data, is_image, mime, language, agent)
    logger.info("upload-report: analysed %d parameters (%s).", len(result["parameters"]), language)
    return AnalysisResponse(**result)


# ---------------------------------------------------------------------------
# POST /api/v1/analyze-trends — compare two reports
# ---------------------------------------------------------------------------

@router.post(
    "/analyze-trends",
    response_model=TrendAnalysisResponse,
    summary="Upload 2 lab reports and get trend analysis",
    description=(
        "Upload an older and a newer report. Both are extracted and classified, "
        "then compared per parameter: direction, absolute change, % change."
    ),
)
async def analyze_trends_endpoint(
    older_report: UploadFile = File(..., description="The OLDER lab report"),
    newer_report: UploadFile = File(..., description="The NEWER lab report"),
    language: str = Depends(validate_language),  # accepted for client compatibility
) -> TrendAnalysisResponse:
    older_args = await _read_upload(older_report)
    newer_args = await _read_upload(newer_report)

    older = await run_in_threadpool(_extract_and_classify, *older_args)
    newer = await run_in_threadpool(_extract_and_classify, *newer_args)

    trend = analyze_trends(older, newer)
    trend["graph_base64"] = await run_in_threadpool(
        generate_trend_graph_base64, trend.get("trends", [])
    )
    logger.info("analyze-trends: compared %d parameters.", len(trend.get("trends", [])))
    return TrendAnalysisResponse(**trend)


# ---------------------------------------------------------------------------
# POST /api/v1/tts — text-to-speech
# ---------------------------------------------------------------------------

@router.post(
    "/tts",
    summary="Generate text-to-speech audio for the summary",
    description="Accepts text and a language name, returns an MP3 audio stream (gTTS).",
)
def generate_tts(request: schemas.TTSRequest) -> StreamingResponse:
    # Plain `def`: FastAPI runs it in a worker thread, so gTTS's blocking
    # network call does not stall the event loop.
    lang_code = SUPPORTED_LANGUAGES.get(request.language)
    if lang_code is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Unsupported language '{request.language}'.")

    try:
        audio = io.BytesIO()
        gTTS(text=request.text, lang=lang_code, slow=False).write_to_fp(audio)
        audio.seek(0)
    except Exception as exc:
        logger.error("TTS generation failed: %s", exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, detail="Failed to generate audio. Please try again."
        ) from exc

    return StreamingResponse(audio, media_type="audio/mpeg")
