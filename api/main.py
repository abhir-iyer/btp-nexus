"""
BTP-Nexus FastAPI Wrapper
Exposes the agentic pipeline as a REST API service.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from nexus_pipeline import run_nexus_pipeline

# ── App Setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="BTP-Nexus API",
    description="Agentic Entity-Resolution & Knowledge-Graph Service for SAP BTP",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response Models ──────────────────────────────────────────────
class ArtifactRequest(BaseModel):
    text: str
    artifact_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Jane confirmed Project Orion budget is 1.2M under BTP cost centre CC-7740.",
                "artifact_id": "TRANSCRIPT-2024-001"
            }
        }

class ArtifactResponse(BaseModel):
    artifact_id: str
    accepted_entities: list
    quarantined_entities: list
    conflict_entities: list
    relationships: list
    graph_write_status: str
    resolver_loops_used: int
    mlflow_run_id: Optional[str]
    coref_text: Optional[str] = None   # coreference-resolved text for debugging

# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "BTP-Nexus",
        "version": "1.0.0",
        "status": "running",
        "description": "Agentic Entity-Resolution & Knowledge-Graph Service",
        "endpoints": {
            "POST /resolve": "Run the full agentic pipeline on an artifact",
            "GET /health":   "Health check",
            "GET /docs":     "Interactive API documentation"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "service": "BTP-Nexus"}

@app.post("/resolve", response_model=ArtifactResponse)
def resolve(req: ArtifactRequest):
    """
    Run the full BTP-Nexus agentic pipeline on a text artifact.

    - De-noises and normalises the raw text
    - Extracts business entities via fine-tuned LLM
    - Resolves entities against the Master Data Graph (Neo4j)
    - Applies 70/30 Consensus Logic with autonomous loop-back
    - Writes reconciled entities to the Knowledge Graph
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
        quarantined_entities=result.quarantined_entities,
        conflict_entities=result.extracted_entities,
        relationships=result.extracted_relationships,
        graph_write_status=result.graph_write_status,
        resolver_loops_used=result.resolver_loop_count,
        mlflow_run_id=result.mlflow_run_id,
        coref_text=result.coref_text,
    )

# ── Run directly ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
