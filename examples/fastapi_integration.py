"""Minimal FastAPI server wrapping mosaicx.sdk -- reference example.

This is a TEACHING EXAMPLE, not production code. It demonstrates how
to wrap the MOSAICX SDK in a web API. For production deployments:

- Add authentication and authorization
- Add rate limiting and request validation
- Use a task queue (Celery, ARQ) for batch processing
- Add structured logging and error tracking
- Run with multiple uvicorn workers

Usage:
    pip install fastapi uvicorn python-multipart
    uvicorn examples.fastapi_integration:app --reload

The server runs at http://localhost:8000 with auto-generated docs
at http://localhost:8000/docs (Swagger UI).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from mosaicx import sdk

app = FastAPI(
    title="MOSAICX API",
    description="Reference API wrapping the MOSAICX medical document structuring SDK.",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class ExtractRequest(BaseModel):
    text: str
    template: str | None = None
    mode: str = "auto"
    score: bool = False


class DeidentifyRequest(BaseModel):
    text: str
    mode: str = "remove"


class SummarizeRequest(BaseModel):
    reports: list[str]
    patient_id: str = "unknown"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    """Check MOSAICX configuration and available capabilities."""
    return sdk.health()


@app.post("/extract")
def extract_text(req: ExtractRequest):
    """Extract structured data from document text."""
    try:
        return sdk.extract(
            req.text, template=req.template, mode=req.mode, score=req.score,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/extract/file")
async def extract_file(
    file: UploadFile,
    template: str | None = None,
    mode: str = "auto",
    score: bool = False,
):
    """Upload a document file (PDF, image, text) and extract structured data."""
    content = await file.read()
    try:
        return sdk.extract(
            documents=content,
            filename=file.filename,
            template=template,
            mode=mode,
            score=score,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/deidentify")
def deidentify(req: DeidentifyRequest):
    """Remove Protected Health Information from text."""
    try:
        return sdk.deidentify(req.text, mode=req.mode)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    """Summarize multiple clinical reports into a patient timeline."""
    try:
        return sdk.summarize(req.reports, patient_id=req.patient_id)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/templates")
def templates():
    """List available extraction templates."""
    return sdk.list_templates()


@app.get("/modes")
def modes():
    """List available extraction modes."""
    return sdk.list_modes()
