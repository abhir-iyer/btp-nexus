"""
BTP-Nexus FastAPI Wrapper
Exposes the agentic pipeline as a REST API service.
"""

import sys
import os

# Ensure root project path is available when running inside Railway
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from nexus_pipeline import run_nexus_pipeline


# ───────────────────────────────────────────────────────────────────────────
# App Setup
# ───────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BTP-Nexus API",
    description="Agentic Entity-Resolution & Knowledge-Graph Service",
    version="2.0.0",
)

# 🔥 Correct CORS configuration for GitHub Pages → Railway communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to your GitHub Pages URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ───────────────────────────────────────────────────────────────────────────

class ArtifactRequest(BaseModel):
    text: str
    artifact_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Finance approved $15M for Phoenix Upgrade.",
                "artifact_id": "TRANSCRIPT-2024-001"
            }
        }


class ArtifactResponse(BaseModel):
    artifact_id: str
    accepted_entities: list
    conflict_entities: list
    quarantined_entities: list
    graph_write_status: str
    resolver_loops_used: int
    mlflow_run_id: Optional[str]


# ───────────────────────────────────────────────────────────────────────────
# Routes
# ───────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "BTP-Nexus",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "POST /resolve": "Run the full agentic pipeline on an artifact",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation"
        }
    }


@app.get("/health")
def health():
    return {"status": "healthy", "service": "BTP-Nexus"}


@app.post("/resolve", response_model=ArtifactResponse)
def resolve(req: ArtifactRequest):
    """
    Run the full BTP-Nexus agentic pipeline on a text artifact.
    """

    if not req.text or len(req.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text is too short to process.")

    try:
        result = run_nexus_pipeline(req.text, artifact_id=req.artifact_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    return ArtifactResponse(
        artifact_id=result.artifact_id,
        accepted_entities=result.resolved_entities,
        conflict_entities=result.extracted_entities,
        quarantined_entities=result.quarantined_entities,
        graph_write_status=result.graph_write_status,
        resolver_loops_used=result.resolver_loop_count,
        mlflow_run_id=result.mlflow_run_id,
    )


# ───────────────────────────────────────────────────────────────────────────
# Run (Railway-Compatible)
# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Railway injects PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
