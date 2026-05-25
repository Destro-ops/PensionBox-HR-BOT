"""
api/main.py

FastAPI server with endpoints:
  GET  /health     — check if the server and index are ready
  POST /ask        — used by the Slack bot to get answers
  POST /reindex    — rebuilds the FAISS index after adding new docs
"""

import logging
import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from api.rag import rag_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up — loading RAG engine...")
    try:
        rag_engine.load()
    except FileNotFoundError as e:
        log.warning(f"RAG engine not ready: {e}")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="PensionBox HR Bot API",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str
    employee_name: str = "Employee"


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class ReindexResponse(BaseModel):
    status: str
    message: str


@app.get("/health")
def health():
    loaded = rag_engine._loaded
    chunks = rag_engine._index.ntotal if loaded and rag_engine._index else 0
    return {
        "status": "ok",
        "rag_loaded": loaded,
        "chunks_indexed": chunks,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not rag_engine._loaded:
        raise HTTPException(
            status_code=503,
            detail="RAG engine not ready. Run `python -m ingest.build_index` first.",
        )

    try:
        result = rag_engine.answer(req.question, req.employee_name)
        return AskResponse(**result)
    except Exception as e:
        log.error(f"Error answering: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")


@app.post("/reindex", response_model=ReindexResponse)
def reindex():
    """Rebuild the FAISS index from the /docs folder. Call after adding new documents."""
    try:
        log.info("Reindex triggered...")
        result = subprocess.run(
            ["python", "-m", "ingest.build_index"],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Reindex failed: {result.stderr}")

        rag_engine._loaded = False
        rag_engine.load()

        return ReindexResponse(
            status="success",
            message=f"Index rebuilt. {rag_engine._index.ntotal} chunks indexed.",
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Reindex timed out.")
    except Exception as e:
        log.error(f"Reindex error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
