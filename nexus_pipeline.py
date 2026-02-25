"""
BTP-Nexus: Agentic Entity-Resolution & Knowledge-Graph Service
LangGraph State Machine — Core Boilerplate
=============================================================
Architecture: Multi-agent pipeline with Consensus-Logic loop-back.
Stack: LangGraph · LangChain · Neo4j · MLflow · Mistral-7B (PEFT/LoRA)
Author: BTP-Nexus Team | SAP BTP Foundational Services
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Optional

import mlflow
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("btp_nexus")


# ---------------------------------------------------------------------------
# 1. DATA MODELS
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    PERSON = "Person"
    PRODUCT = "Product"
    PROJECT = "Project"
    BUDGET = "Budget"
    ORGANIZATION = "Organization"
    UNKNOWN = "Unknown"


class ConfidenceTier(str, Enum):
    ACCEPTED = "ACCEPTED"        # >= 0.70
    CONFLICT = "CONFLICT"        # 0.30 – 0.69 → loop back
    QUARANTINED = "QUARANTINED"  # < 0.30 → human review


@dataclass
class ExtractedEntity:
    raw_name: str
    entity_type: EntityType
    source_artifact_id: str
    attributes: dict[str, Any] = field(default_factory=dict)
    canonical_id: Optional[str] = None
    confidence: float = 0.0
    tier: ConfidenceTier = ConfidenceTier.CONFLICT
    loop_count: int = 0


class NexusState(BaseModel):
    """
    LangGraph shared state object — the single source of truth
    flowing through every node in the pipeline.
    """
    # Raw inputs
    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_text: str = ""
    cleaned_text: str = ""

    # Agent outputs
    extracted_entities: list[dict] = Field(default_factory=list)
    resolved_entities: list[dict] = Field(default_factory=list)
    quarantined_entities: list[dict] = Field(default_factory=list)
    conflict_entities: list[dict] = Field(default_factory=list)   # unresolved after max loops

    # Resolver bookkeeping
    resolver_loop_count: int = 0
    max_resolver_loops: int = 3

    # Graph write status
    graph_write_status: str = "PENDING"

    # MLflow run context
    mlflow_run_id: Optional[str] = None

    # LangGraph message thread (append-only)
    messages: Annotated[list, add_messages] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. DE-NOISING PIPELINE  (Data Engineering Layer)
# ---------------------------------------------------------------------------

class EntityDenoiser:
    """
    Normalises raw artifact text before LLM extraction.
    Analogous to audio signal de-noising: remove artefacts, standardise signal.
    """

    ALIAS_MAP: dict[str, str] = {
        # Product aliases — extend via SAP MDG export
        "orion v2": "Orion-v2",
        "project orion": "Orion-v2",
        "s/4 hana": "SAP S/4HANA",
        "s4hana": "SAP S/4HANA",
        "hana cloud": "SAP HANA Cloud",
        "btp": "SAP Business Technology Platform",
    }

    def normalize(self, text: str) -> str:
        """Lower-frequency noise removal + entity standardization."""
        import re

        # Step 1 — strip boilerplate headers / footers (simulated)
        text = re.sub(r"(?i)(confidential|internal use only|page \d+ of \d+)", "", text)

        # Step 2 — fix common OCR errors
        text = text.replace("\x00", "").replace("\ufffd", "?")

        # Step 3 — alias resolution (case-insensitive)
        for alias, canonical in self.ALIAS_MAP.items():
            text = re.sub(re.escape(alias), canonical, text, flags=re.IGNORECASE)

        # Step 4 — collapse excess whitespace
        text = re.sub(r"\s+", " ", text).strip()

        logger.info("De-noising complete. Output length: %d chars", len(text))
        return text


# ---------------------------------------------------------------------------
# 3. AGENT NODES
# ---------------------------------------------------------------------------

# --- Shared LLM (swap ChatOpenAI for vLLM endpoint serving Mistral-7B LoRA) ---
LLM = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def node_denoise(state: NexusState, config: RunnableConfig) -> dict:
    """Node 0 — Data Engineering: clean raw artifact text."""
    denoiser = EntityDenoiser()
    cleaned = denoiser.normalize(state.raw_text)
    return {"cleaned_text": cleaned}


EXTRACTOR_SYSTEM_PROMPT = """\
You are an SAP Enterprise Entity Extractor.
Given a business artifact, extract ALL entities into strict JSON.
Entity types: Person, Product, Project, Budget, Organization.

CRITICAL RULES:
- Person: extract the person's name only (e.g. "Jane", "Marco Rossi")
- Project: extract the project name only (e.g. "Orion-v2", "Project Titan")
- Product: extract the product/system name (e.g. "SAP S/4HANA", "HANA Cloud")
- Organization: extract the org/department name (e.g. "Finance", "SAP SE")
- Budget: extract ONLY the actual monetary value or cost centre code as written
    (e.g. "$38M", "$42M", "CC-7740", "USD 1.2M")
    Do NOT extract descriptive phrases like "Orion budget" or "project allocation".
    If a sentence mentions multiple amounts, extract each as a separate Budget entity.

Return ONLY valid JSON in this exact schema (no markdown fences, no prose):
{
  "entities": [
    {
      "raw_name": "<string>",
      "entity_type": "<Person|Product|Project|Budget|Organization>",
      "attributes": { "<key>": "<value>" }
    }
  ]
}
"""


def node_extractor(state: NexusState, config: RunnableConfig) -> dict:
    """Node 1 — Extractor Agent: pulls typed entities from cleaned text via LLM."""
    logger.info("[Extractor] Extracting entities from cleaned text.")

    messages = [
        SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
        HumanMessage(content=f"Extract all entities from this artifact:\n\n{state.cleaned_text}"),
    ]

    response = LLM.invoke(messages)

    try:
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
        payload = json.loads(raw)
        entities = payload.get("entities", [])
    except (json.JSONDecodeError, Exception):
        logger.warning("[Extractor] JSON parse failed — returning empty entity list.")
        entities = []

    # Post-extraction normalisation:
    # If the LLM returned a descriptive Budget phrase (e.g. "Orion-v2 budget")
    # instead of the actual monetary values, extract the real values from the
    # original text and replace/supplement the vague entity.
    normalised = []
    MONEY_PATTERN = re.compile(
        r'\$\d+(?:\.\d+)?[BMKbmk]?'        # $38M  $42M  $4.2B  $800
        r'|\b\d+(?:\.\d+)?[BMKbmk]\b'      # 38M   1.2B  800K
        r'|\bUSD\s+\d+(?:\.\d+)?[BMKbmk]?' # USD 1.2M
        r'|\bCC-\d{3,6}\b',                 # CC-7740
        re.IGNORECASE
    )

    for ent in entities:
        if ent.get("entity_type") == "Budget":
            raw_name = ent.get("raw_name", "")
            # Check if it looks like a descriptive phrase rather than an actual value
            has_actual_value = bool(re.search(r'[\$£€]|\bCC-|\b\d.*[BMKbmk]\b|\bUSD\b|\bEUR\b', raw_name, re.IGNORECASE))
            if not has_actual_value:
                # Mine the source text for real monetary values instead
                found_values = MONEY_PATTERN.findall(state.cleaned_text)
                if found_values:
                    logger.info("[Extractor] Replacing vague Budget '%s' with actual values: %s", raw_name, found_values)
                    for val in found_values:
                        normalised.append({
                            "raw_name": val.strip(),
                            "entity_type": "Budget",
                            "attributes": {"source_phrase": raw_name},
                        })
                    continue  # skip the original vague entity
        normalised.append(ent)

    logger.info("[Extractor] Found %d entities (%d after normalisation).", len(entities), len(normalised))
    return {"extracted_entities": normalised}


def _query_master_data_graph(entities: list[dict]) -> dict[str, dict]:
    """
    Query Master Data Graph for candidate matches.
    Uses exact → substring → fuzzy fallback so LLM name variants still resolve.
    Returns: { raw_name -> { canonical_id, similarity } }
    """
    import difflib

    MASTER_DATA = {
        # Projects
        "orion-v2":       {"canonical_id": "MDG-PROJ-00142", "similarity": 0.95},
        "project orion":  {"canonical_id": "MDG-PROJ-00142", "similarity": 0.93},
        "orion":          {"canonical_id": "MDG-PROJ-00142", "similarity": 0.87},
        # Products / Tech
        "sap s/4hana":    {"canonical_id": "MDG-PROD-00001", "similarity": 0.99},
        "sap btp":        {"canonical_id": "MDG-PROD-00010", "similarity": 0.99},
        "hana cloud":     {"canonical_id": "MDG-PROD-00011", "similarity": 0.97},
        # People
        "jane smith":     {"canonical_id": "MDG-PERSON-0421", "similarity": 0.92},
        "jane":           {"canonical_id": "MDG-PERSON-0421", "similarity": 0.74},
        "marco":          {"canonical_id": "MDG-PERSON-0388", "similarity": 0.71},
        "marco rossi":    {"canonical_id": "MDG-PERSON-0388", "similarity": 0.91},
        "priya":          {"canonical_id": "MDG-PERSON-0305", "similarity": 0.72},
        "priya k.":       {"canonical_id": "MDG-PERSON-0305", "similarity": 0.85},
        "sarah chen":     {"canonical_id": "MDG-PERSON-0512", "similarity": 0.94},
        # Organisations
        "finance":        {"canonical_id": "MDG-ORG-00204",  "similarity": 0.78},
        "sap se":         {"canonical_id": "MDG-ORG-00001",  "similarity": 0.99},
        "titancorp":      {"canonical_id": "MDG-ORG-00318",  "similarity": 0.90},
        "cloudbase inc":  {"canonical_id": "MDG-ORG-00401",  "similarity": 0.88},
        # Budgets / Cost Centres
        "cc-7740":        {"canonical_id": "MDG-CC-7740",    "similarity": 0.99},
        "cc-9981":        {"canonical_id": "MDG-CC-9981",    "similarity": 0.99},
        "cc-5510":        {"canonical_id": "MDG-CC-5510",    "similarity": 0.99},
        "cc-8821":        {"canonical_id": "MDG-CC-8821",    "similarity": 0.99},
        "$38m":           {"canonical_id": "MDG-BUDG-0831",  "similarity": 0.82},
        "$42m":           {"canonical_id": "MDG-BUDG-0842",  "similarity": 0.82},
        "$4m":            {"canonical_id": "MDG-BUDG-0804",  "similarity": 0.75},
        "usd 1.2m":       {"canonical_id": "MDG-BUDG-0120",  "similarity": 0.84},
        "$4.2b":          {"canonical_id": "MDG-BUDG-0420",  "similarity": 0.80},
        "1.2m":           {"canonical_id": "MDG-BUDG-0120",  "similarity": 0.80},
        "800k":           {"canonical_id": "MDG-BUDG-0800",  "similarity": 0.78},
    }

    mdg_keys = list(MASTER_DATA.keys())
    results = {}

    for ent in entities:
        raw = ent["raw_name"]
        key = raw.lower().strip()

        # 1. Exact match
        if key in MASTER_DATA:
            results[raw] = MASTER_DATA[key]
            continue

        # 2. Substring match — key appears inside the extracted name or vice versa
        #    e.g. "$42M - $38M" contains "$42m" after lowercasing
        substring_hit = None
        substring_score = 0.0
        for mk in mdg_keys:
            if mk in key or key in mk:
                # prefer the master key with the higher base similarity
                candidate_score = MASTER_DATA[mk]["similarity"] * 0.90  # small penalty for non-exact
                if candidate_score > substring_score:
                    substring_score = candidate_score
                    substring_hit = MASTER_DATA[mk].copy()
                    substring_hit["similarity"] = round(candidate_score, 3)
        if substring_hit:
            results[raw] = substring_hit
            continue

        # 3. Fuzzy match — catch typos and LLM paraphrases
        #    e.g. "Marco Rossi" → "marco rossi" (0.95 ratio)
        close = difflib.get_close_matches(key, mdg_keys, n=1, cutoff=0.60)
        if close:
            ratio = difflib.SequenceMatcher(None, key, close[0]).ratio()
            hit = MASTER_DATA[close[0]].copy()
            hit["similarity"] = round(hit["similarity"] * ratio, 3)
            results[raw] = hit
            continue

        # 4. No match
        results[raw] = {"canonical_id": None, "similarity": 0.0}

    return results


def node_resolver(state: NexusState, config: RunnableConfig) -> dict:
    """
    Node 2 — Forensic Resolver.
    Uses master-data similarity scores directly (deterministic 70/30 logic).
    The LLM is used only for entity extraction; scoring is deterministic.
    """
    logger.info(
        "[Resolver] Loop %d / %d",
        state.resolver_loop_count + 1,
        state.max_resolver_loops,
    )

    # Type-based fallback scores for entities not in master data.
    # Reflects how identifiable an entity type is without an exact MDG record.
    TYPE_FALLBACK_SCORE = {
        "Budget":       0.72,   # monetary values are self-identifying
        "Person":       0.55,   # single names are ambiguous → conflict tier
        "Project":      0.50,   # project names need corroboration
        "Product":      0.45,   # unknown products need review
        "Organization": 0.48,
        "Organisation": 0.48,
        "Unknown":      0.20,
    }

    master_data = _query_master_data_graph(state.extracted_entities)

    accepted, conflicts, quarantined = [], [], []

    for ent in state.extracted_entities:
        candidate = master_data.get(ent["raw_name"], {})
        similarity = float(candidate.get("similarity", 0.0))
        canonical_id = candidate.get("canonical_id")

        # If no MDG match, use entity-type fallback so valid new entities
        # land in CONFLICT (for loop-back review) rather than auto-quarantine
        if similarity == 0.0:
            entity_type = ent.get("entity_type", ent.get("type", "Unknown"))
            similarity = TYPE_FALLBACK_SCORE.get(entity_type, 0.20)
            canonical_id = None  # no canonical ID yet — needs human review

        ent["confidence"] = similarity
        ent["canonical_id"] = canonical_id
        ent["loop_count"] = state.resolver_loop_count

        # 70/30 Consensus Logic — fully deterministic from MDG similarity
        if similarity >= 0.70:
            ent["tier"] = ConfidenceTier.ACCEPTED
            accepted.append(ent)
        elif similarity >= 0.30:
            ent["tier"] = ConfidenceTier.CONFLICT
            conflicts.append(ent)
        else:
            ent["tier"] = ConfidenceTier.QUARANTINED
            quarantined.append(ent)

    avg_confidence = (
        sum(e["confidence"] for e in accepted + conflicts + quarantined)
        / max(len(state.extracted_entities), 1)
    )

    if state.mlflow_run_id:
        with mlflow.start_run(run_id=state.mlflow_run_id, nested=True):
            mlflow.log_metric("avg_reconciliation_confidence", avg_confidence, step=state.resolver_loop_count)
            mlflow.log_metric("accepted_count",    len(accepted),    step=state.resolver_loop_count)
            mlflow.log_metric("conflict_count",    len(conflicts),   step=state.resolver_loop_count)
            mlflow.log_metric("quarantined_count", len(quarantined), step=state.resolver_loop_count)

    logger.info(
        "[Resolver] Accepted=%d | Conflicts=%d | Quarantined=%d",
        len(accepted), len(conflicts), len(quarantined),
    )

    # If this is the final loop, move remaining conflicts to conflict_entities
    # so they're surfaced in the API response rather than silently dropped
    is_final_loop = (state.resolver_loop_count + 1) >= state.max_resolver_loops
    final_conflicts = []
    if is_final_loop and conflicts:
        for ent in conflicts:
            ent["tier"] = ConfidenceTier.CONFLICT
        final_conflicts = conflicts
        conflicts = []  # clear so router sees no extracted_entities → proceeds to graph_architect

    return {
        "resolved_entities": accepted,
        "extracted_entities": conflicts,
        "quarantined_entities": state.quarantined_entities + quarantined,
        "conflict_entities": state.conflict_entities + final_conflicts,
        "resolver_loop_count": state.resolver_loop_count + 1,
    }


def node_graph_architect(state: NexusState, config: RunnableConfig) -> dict:
    """
    Node 3 — Graph Architect Agent.
    Writes reconciled entity triples into Neo4j Knowledge Graph.
    """
    logger.info("[GraphArchitect] Writing %d entities to Knowledge Graph.", len(state.resolved_entities))

    # --- Neo4j Write (simulated; replace URI/auth for real instance) ---
    # --- Neo4j — reads from environment (AuraDB cloud or local Docker) ---
    NEO4J_URI  = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
    NEO4J_USER = os.environ.get("NEO4J_USER",     "neo4j")
    NEO4J_PASS = os.environ.get("NEO4J_PASSWORD", "password")
    NEO4J_AUTH = (NEO4J_USER, NEO4J_PASS)

    CYPHER_MERGE = """\
    MERGE (e:Entity {canonical_id: $canonical_id})
    ON CREATE SET
        e.raw_name = $raw_name,
        e.entity_type = $entity_type,
        e.confidence = $confidence,
        e.created_at = datetime(),
        e.source_artifact = $source_artifact_id
    ON MATCH SET
        e.last_seen = datetime(),
        e.confidence = CASE WHEN $confidence > e.confidence
                           THEN $confidence ELSE e.confidence END
    """

    write_results = []
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        with driver.session() as session:
            for ent in state.resolved_entities:
                if ent.get("canonical_id"):
                    session.run(CYPHER_MERGE, **ent)
                    write_results.append(ent["canonical_id"])
        driver.close()
        status = f"SUCCESS: wrote {len(write_results)} nodes"
    except Exception as exc:
        logger.warning("[GraphArchitect] Neo4j unavailable (simulation mode): %s", exc)
        status = f"SIMULATED: would write {len(state.resolved_entities)} nodes"

    logger.info("[GraphArchitect] Status: %s", status)
    return {"graph_write_status": status}


# ---------------------------------------------------------------------------
# 4. ROUTING LOGIC
# ---------------------------------------------------------------------------

def route_resolver(state: NexusState) -> str:
    """
    Consensus Logic Router — pure routing only, no state mutation.
    - Conflicts remain AND loops left → re-run resolver
    - Max loops hit OR no conflicts → proceed to Graph Architect
    - Nothing accepted → end
    """
    has_conflicts  = len(state.extracted_entities) > 0
    loop_exhausted = state.resolver_loop_count >= state.max_resolver_loops
    has_accepted   = len(state.resolved_entities) > 0

    if has_conflicts and not loop_exhausted:
        logger.info("[Router] Conflicts remain. Looping back (loop %d).", state.resolver_loop_count)
        return "resolver"

    if has_accepted:
        logger.info("[Router] Routing to Graph Architect.")
        return "graph_architect"

    logger.warning("[Router] No accepted entities. Ending pipeline.")
    return END


# ---------------------------------------------------------------------------
# 5. LANGGRAPH STATE MACHINE ASSEMBLY
# ---------------------------------------------------------------------------

def build_nexus_graph() -> StateGraph:
    """
    Assembles the BTP-Nexus LangGraph state machine.

    Flow:
        START → denoise → extractor → resolver ⟲ (up to N loops)
                                          └─ graph_architect → END
                                          └─ END (all quarantined)
    """
    graph = StateGraph(NexusState)

    # Register nodes
    graph.add_node("denoise", node_denoise)
    graph.add_node("extractor", node_extractor)
    graph.add_node("resolver", node_resolver)
    graph.add_node("graph_architect", node_graph_architect)

    # Define edges
    graph.add_edge(START, "denoise")
    graph.add_edge("denoise", "extractor")
    graph.add_edge("extractor", "resolver")

    # Conditional loop-back from resolver
    graph.add_conditional_edges(
        "resolver",
        route_resolver,
        {
            "resolver": "resolver",
            "graph_architect": "graph_architect",
            END: END,
        },
    )
    graph.add_edge("graph_architect", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# 6. MLOPS WRAPPER — MLflow Experiment Tracking
# ---------------------------------------------------------------------------

def run_nexus_pipeline(raw_text: str, artifact_id: Optional[str] = None) -> NexusState:
    """
    Entry point for the BTP-Nexus pipeline.
    Initialises MLflow tracking run, executes LangGraph, returns final state.
    """
    mlflow.set_experiment("BTP-Nexus-Entity-Resolution")

    with mlflow.start_run() as run:
        mlflow.set_tags({
            "service": "BTP-Nexus",
            "version": "1.0.0",
            "artifact_id": artifact_id or "unknown",
        })
        mlflow.log_param("max_resolver_loops", 3)
        mlflow.log_param("confidence_accept_threshold", 0.70)
        mlflow.log_param("confidence_quarantine_threshold", 0.30)

        initial_state = NexusState(
            raw_text=raw_text,
            artifact_id=artifact_id or str(uuid.uuid4()),
            mlflow_run_id=run.info.run_id,
        )

        nexus_graph = build_nexus_graph()
        final_state_dict = nexus_graph.invoke(initial_state)
        final_state = NexusState(**final_state_dict)

        # Final summary metrics
        total = (
            len(final_state.resolved_entities)
            + len(final_state.extracted_entities)  # leftover conflicts
            + len(final_state.quarantined_entities)
        )
        mlflow.log_metric("total_entities_processed", total)
        mlflow.log_metric("final_accepted", len(final_state.resolved_entities))
        mlflow.log_metric("final_quarantined", len(final_state.quarantined_entities))
        mlflow.log_metric(
            "acceptance_rate",
            len(final_state.resolved_entities) / max(total, 1),
        )
        mlflow.log_text(final_state.graph_write_status, "graph_write_status.txt")

    logger.info("[Pipeline] Complete. Run ID: %s", run.info.run_id)
    return final_state


# ---------------------------------------------------------------------------
# 7. FINE-TUNING CONFIGURATION STUB  (PEFT / LoRA)
# ---------------------------------------------------------------------------
# Full training script lives in /training/finetune_mistral.py
# This stub documents key hyperparameters for the LoRA adapter.

LORA_CONFIG = {
    "base_model": "mistralai/Mistral-7B-v0.1",
    "task_type": "CAUSAL_LM",
    "lora_r": 16,                  # Rank of update matrices
    "lora_alpha": 32,              # Scaling factor
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],   # Attention layers only
    "bias": "none",
    "training_dataset": "data/sap_entity_resolution_v1.jsonl",
    "max_seq_length": 2048,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "fp16": True,
    "output_dir": "checkpoints/btp-nexus-mistral-lora",
    "mlflow_tracking_uri": "http://mlflow-server:5000",
    "mlflow_experiment": "BTP-Nexus-LoRA-Finetune",
}
"""
Training data format (SAP entity resolution pairs):
{"prompt": "Resolve: 'Project Orion' | Master Data: [Orion-v2, Orion-Classic]",
 "completion": '{"canonical_id": "MDG-PROD-00142", "confidence": 0.92}'}
{"prompt": "Resolve: 'Jane S.' | Master Data: [Jane Smith (HR-DE), Jane Simpson (FI-US)]",
 "completion": '{"canonical_id": "MDG-PERSON-0421", "confidence": 0.78}'}
"""

# ---------------------------------------------------------------------------
# 8. DEMO ENTRYPOINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE_ARTIFACT = """
    INTERNAL USE ONLY  —  Project Kick-off Meeting Transcript  —  Page 1 of 3

    Attendees: Jane S. (Product Lead), Marco Rossi (Finance), Priya K. (Engineering)

    Jane confirmed that Project Orion is now officially renamed Orion v2 and will run
    on S4HANA with HANA Cloud integration. Marco noted the approved budget is USD 1.2M
    under SAP BTP cost center CC-7740. Priya said the first sprint for orion-v2
    kicks off Monday. Jane Smith will own the roadmap sign-off.

    CONFIDENTIAL
    """

    result = run_nexus_pipeline(SAMPLE_ARTIFACT, artifact_id="TRANSCRIPT-2024-001")

    print("\n" + "="*60)
    print("BTP-NEXUS PIPELINE RESULTS")
    print("="*60)
    print(f"✅ Accepted Entities  : {len(result.resolved_entities)}")
    print(f"⚠️  Conflicts Remaining: {len(result.extracted_entities)}")
    print(f"🚨 Quarantined         : {len(result.quarantined_entities)}")
    print(f"🗄️  Graph Write Status : {result.graph_write_status}")
    print(f"🔄 Resolver Loops Used : {result.resolver_loop_count}")
