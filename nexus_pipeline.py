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
    CONDITION = "Condition"
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
    coref_text: str = ""   # coreference-resolved + normalised text fed to extractor

    # Agent outputs
    extracted_entities: list[dict] = Field(default_factory=list)
    resolved_entities: list[dict] = Field(default_factory=list)
    quarantined_entities: list[dict] = Field(default_factory=list)

    # Resolver bookkeeping
    resolver_loop_count: int = 0
    max_resolver_loops: int = 3

    # LLM-extracted relationships (evidence-backed triples)
    extracted_relationships: list[dict] = Field(default_factory=list)

    # Fingerprint of conflict entity names from the previous loop —
    # used to detect when looping is making no progress
    last_conflict_fingerprint: str = ""

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


EXTRACTOR_SYSTEM_PROMPT = """\You are an enterprise entity extractor for a knowledge graph.

TASK: Read the text and extract every distinct named business entity.
Think step-by-step before outputting. First identify candidates, then classify.

ENTITY TYPES — definitions and decision rules:

  Person        Any individual human referenced anywhere in the text — in the body,
                greeting line ("Hi X", "Dear X"), sign-off ("Regards, X", "— X"),
                CC field, or as a quoted speaker. Use the name exactly as it appears.

  Project       Any named unit of work, initiative, programme, or investment. The
                trigger does NOT need to be the word "Project". Classify as Project
                if the name refers to a bounded effort with a scope or budget.
                Signals: Upgrade / Rollout / Migration / Modernization / Analytics /
                Programme / Program / Initiative / Phase / Refresh / Launch /
                Transformation / Deployment / Integration / Implementation / X / v2 /
                Plus — OR any multi-word capitalised noun phrase that is not a person
                or organisation.
                DISAMBIGUATION: two names that differ by even one word or character
                (e.g. "Horizon Modernization" vs "Horizon Analytics", "Epsilon" vs
                "Epsilon-X") are SEPARATE entities. Extract each in full.

  Product       Named software system, platform, tool, or technology product.

  Organization  Any department, division, team, committee, company, vendor, regulator,
                or governance body that acts as an agent. This includes lowercase
                functional references when they act as a decision-maker or gate:
                "finance", "compliance", "legal", "treasury", "procurement", "board",
                "audit". Capitalise the first letter in raw_name.

  Budget        A literal monetary value or cost-centre code exactly as written.
                "$28M", "CC-7740", "USD 1.2M". One entity per distinct amount.
                Never extract descriptive phrases.

  Condition     A named prerequisite, gate, clearance, or sign-off that must occur
                before a project or budget can proceed.
                e.g. "cybersecurity clearance", "compliance sign-off", "board approval",
                "regulatory clearance". Extract even if lowercase.

EXTRACTION PRINCIPLES — apply to every input regardless of phrasing:

  1. SCAN ALL ZONES: greetings, sign-offs, body, footnotes, bullet points, headers.
     Names in "Hi X" and "Thanks, X" are real entities.

  2. NEGATED CONTEXTS: An entity that EXISTS but is being EXCLUDED or DENIED
     something should still be extracted.
       "The $28M does not apply to Horizon Analytics" →
           extract "Horizon Analytics" (it exists — the exclusion is a relationship)
     Only skip entities that are explicitly stated to NOT EXIST:
       "no formal memo was issued" → do NOT extract "formal memo"

  3. CONDITIONAL CONTEXTS: Extract the entity even if its state is conditional.
       "subject to cybersecurity clearance" →
           extract "cybersecurity clearance" as Condition

  4. HEDGE CONTEXTS: Extract entities even in proposed / in-principle statements.
       "Finance agreed in principle to fund Epsilon" →
           extract Finance, Epsilon — the hedge is a relationship qualifier

  5. LIBERAL BIAS: When uncertain, extract and let the resolver score it.
     False positives are filtered downstream. False negatives are permanent losses.

Return ONLY valid JSON — no markdown fences, no prose, no explanation:
{
  "entities": [
    {
      "raw_name": "<exact string from text>",
      "entity_type": "<Person|Product|Project|Budget|Organization|Condition>",
      "attributes": {}
    }
  ]
}
"""


def _robust_parse_json(raw: str, label: str) -> dict:
    """
    Multi-strategy JSON extractor — never silently returns {}.
    Tries in order:
      1. Direct parse (model obeyed instructions)
      2. Strip all markdown fences (``` or ```json) anywhere in the string
      3. Extract first {...} block via regex (handles prose wrapping)
      4. Extract first [...] block (in case model returned a bare array)
    Logs the raw response on failure so the problem is visible in docker logs.
    """
    text = raw.strip()

    # Strategy 1 — direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2 — strip markdown fences anywhere in the string
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3 — pull out the first complete {...} block
    brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 4 — bare array fallback (model returned [...] without wrapper)
    bracket_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if bracket_match:
        try:
            return {"entities": json.loads(bracket_match.group())}
        except json.JSONDecodeError:
            pass

    # All strategies failed — log raw so the problem is visible
    logger.error(
        "[%s] All JSON parse strategies failed.\n"
        "=== RAW LLM RESPONSE ===\n%s\n========================",
        label, raw[:2000]
    )
    return {}


# ---------------------------------------------------------------------------
# 2b. COREFERENCE RESOLUTION NODE
# ---------------------------------------------------------------------------

COREF_SYSTEM_PROMPT = """\You are a text pre-processor for an enterprise knowledge graph pipeline.
Your only job is to rewrite the input text to make entity extraction easier.
Do NOT summarize, analyze, or change meaning.

REWRITE RULES (apply all that are relevant):

AMOUNTS: Convert any informal or written-out number that refers to money into
canonical $NM / $NB form. Context clues: near words like "budget", "cost",
"allocation", "fund", "spend", "price", "fee", "investment".
  "eight million"          → $8M
  "the original eight"     → $8M   (if prior context established $8M)
  "ten flat"               → $10M
  "a billion and a half"   → $1.5B
  "three point two"        → $3.2M (if context = millions)
  Do NOT convert years, quantities, counts.

COREFERENCE: Replace pronouns and vague references with their antecedent.
  "the project"     → the actual project name if clear from context
  "the fund"        → the actual budget amount
  "it" / "they"     → the antecedent entity
  "the same amount" → the actual dollar figure
  If the referent is ambiguous, leave the original text unchanged.

PRESERVE EXACTLY — do not rephrase or remove:
  - Conditional language: "subject to", "pending", "contingent on"
  - Negation: "no X was issued", "does not apply to", "not approved"
  - Hedges: "agreed in principle", "in discussion", "proposed"
  - Distinctions: "X is separate from Y", "X ≠ Y", "X remains distinct"

Return ONLY the rewritten text. No explanation, no JSON.
"""

COREF_EXAMPLES = [
    (
        "Epsilon Rollout was approved at $8M. Later discussion referenced \"the original eight\" as insufficient, proposing \"ten flat\".",
        "Epsilon Rollout was approved at $8M. Later discussion referenced $8M as insufficient, proposing $10M."
    ),
    (
        "The board approved it. Finance agreed in principle, though no formal memo was issued.",
        "The board approved it. Finance agreed in principle, though no formal memo was issued."
    ),
]


def node_coreference(state: NexusState, config: RunnableConfig) -> dict:
    """
    Node 0b — Coreference Resolution & Text Normalisation.
    Rewrites informal amounts, resolves anaphora, preserves negations.
    Feeds coref_text to the extractor instead of raw cleaned_text.
    """
    logger.info("[Coref] Resolving coreferences and normalising amounts.")

    few_shot = "\n\n".join(
        f"INPUT:\n{inp}\n\nOUTPUT:\n{out}"
        for inp, out in COREF_EXAMPLES
    )

    messages = [
        SystemMessage(content=COREF_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"EXAMPLES:\n\n{few_shot}\n\n"
            f"Now rewrite this text:\n\nINPUT:\n{state.cleaned_text}\n\nOUTPUT:"
        )),
    ]

    response = LLM.invoke(messages)
    coref_text = response.content.strip()

    logger.info("[Coref] Original : %s", state.cleaned_text[:200])
    logger.info("[Coref] Resolved : %s", coref_text[:200])

    return {"coref_text": coref_text}


def _clean_entity_list(entities: list[dict]) -> list[dict]:
    """
    Post-extraction cleanup — runs after self-critique merges recovered entities.

    Four passes, in order:

    1. ACCOUNT-CODE RECLASSIFICATION
       OPEX, CAPEX and similar account classification codes are NOT budget amounts —
       they are accounting categories. LLMs routinely mislabel them as Budget because
       they appear near money language. Force-correct to Organisation type.

    2. FRAGMENT FILTER
       Drop entities whose raw_name starts with a logical operator or article,
       indicating a clause fragment rather than a real entity.
       e.g. "if clearance" → drop (trailing clause from "if clearance is denied")
            "the remaining" → drop,  "when approved" → drop

    3. CROSS-TYPE SUBSTRING DEDUP
       If entity A's name is a pure substring of entity B's name — regardless of
       type — drop A and keep B. This handles the case where the LLM extracts:
         "Helix"  (typed as Person)  +  "Helix Transformation" (Project)
       The per-type version would keep both because they're different types. The
       cross-type version correctly suppresses "Helix".
       Safe guard: only suppress if B is a Project or Organisation (not Budget/
       Condition), so "$3M" inside "$3M contingency" is never suppressed.

    4. WITHIN-TYPE SUBSTRING DEDUP
       Same rule applied within a single type as a final pass, e.g.
       "Finance" (Org) substring-of "Finance Committee" (Org) → keep the longer.
    """

    # ── Pass 1: Account-code reclassification ────────────────────────────
    # OPEX / CAPEX / EBITDA etc. are accounting categories, not budget amounts.
    ACCOUNT_CODES: dict[str, str] = {
        "opex":   "MDG-ACC-OPEX",
        "capex":  "MDG-ACC-CAPEX",
        "ebitda": "MDG-ACC-EBITDA",
        "ebit":   "MDG-ACC-EBIT",
        "cogs":   "MDG-ACC-COGS",
        "sga":    "MDG-ACC-SGA",
    }
    reclassified = []
    for e in entities:
        raw   = e.get("raw_name", "").strip()
        lower = raw.lower()
        if lower in ACCOUNT_CODES and e.get("entity_type") == "Budget":
            e = dict(e)
            e["entity_type"] = "Organisation"
            e["attributes"]  = {**e.get("attributes", {}), "account_code": ACCOUNT_CODES[lower]}
            logger.info("[CleanEntities] Reclassified '%s' Budget → Organisation (account code)", raw)
        reclassified.append(e)

    # ── Pass 2: Fragment filter ───────────────────────────────────────────
    FRAGMENT_PREFIXES = {
        "if ", "when ", "as ", "the ", "a ", "an ", "no ", "not ",
        "any ", "all ", "its ", "this ", "that ", "which ", "once ",
        "until ", "unless ", "before ", "after ", "upon ", "for the ",
        "pending ", "subject to ", "contingent on ",
    }
    filtered = []
    for e in reclassified:
        raw   = e.get("raw_name", "").strip()
        lower = raw.lower()
        if any(lower.startswith(pfx) for pfx in FRAGMENT_PREFIXES):
            logger.info("[CleanEntities] Dropping fragment: '%s'", raw)
            continue
        if len(raw) <= 2:
            logger.info("[CleanEntities] Dropping too-short entity: '%s'", raw)
            continue
        filtered.append(e)

    # ── Pass 3: Cross-type substring dedup ───────────────────────────────
    # Collect all names that appear as a substring of a longer Project or
    # Organisation entity. Drop ANY entity (regardless of its own type)
    # whose name is a substring of one of those longer names.
    # Guard: only suppress against Project/Organisation targets, never against
    # Budget/Condition (prevents "$3M" being swallowed by "$3M contingency").
    ANCHOR_TYPES = {"Project", "Organisation", "Organization"}
    all_names_lower = [e.get("raw_name", "").lower() for e in filtered]
    anchor_names    = [
        e.get("raw_name", "").lower()
        for e in filtered
        if e.get("entity_type", "") in ANCHOR_TYPES
    ]

    cross_deduped = []
    for e in filtered:
        n = e.get("raw_name", "").lower()
        # Check if this name is a strict substring of any anchor name
        swallowed_by = next(
            (a for a in anchor_names if n != a and n in a),
            None
        )
        if swallowed_by:
            logger.info(
                "[CleanEntities] Cross-type drop '%s' (%s) — substring of '%s'",
                e.get("raw_name", ""), e.get("entity_type", ""), swallowed_by
            )
        else:
            cross_deduped.append(e)

    # ── Pass 4: Within-type substring dedup ──────────────────────────────
    by_type: dict[str, list[dict]] = {}
    for e in cross_deduped:
        by_type.setdefault(e.get("entity_type", "Unknown"), []).append(e)

    result = []
    for etype, group in by_type.items():
        names_lower = [e.get("raw_name", "").lower() for e in group]
        for i, e in enumerate(group):
            n = names_lower[i]
            is_substring = any(
                n != names_lower[j] and n in names_lower[j]
                for j in range(len(group))
            )
            if is_substring:
                logger.info(
                    "[CleanEntities] Within-type drop '%s' (superseded by longer %s name)",
                    e.get("raw_name", ""), etype
                )
            else:
                result.append(e)

    return result


def node_extractor(state: NexusState, config: RunnableConfig) -> dict:
    """Node 1 — Extractor Agent: pulls typed entities from cleaned text via LLM."""
    logger.info("[Extractor] Extracting entities from cleaned text.")

    # Use coreference-resolved text if available, fall back to cleaned_text
    source_text = state.coref_text if state.coref_text else state.cleaned_text
    messages = [
        SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
        HumanMessage(content=f"Extract all entities from this artifact:\n\n{source_text}"),
    ]

    response = LLM.invoke(messages)
    logger.debug("[Extractor] Raw response (first 300 chars): %s", response.content[:300])

    payload = _robust_parse_json(response.content, "Extractor")
    entities = payload.get("entities", [])

    if not entities:
        logger.warning(
            "[Extractor] Got 0 entities after parsing. Payload keys: %s",
            list(payload.keys())
        )

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

    # ── FIX 2: Pattern to extract a bare monetary value from a noisy string.
    # Handles LLM outputs like "$24M allocation" or "$21M base amount".
    BARE_MONEY = re.compile(
        r'[\$£€]\d+(?:[.,]\d+)?[BMKbmk]?'
        r'|\bCC-\d{3,6}\b'
        r'|\bUSD\s*\d+(?:[.,]\d+)?[BMKbmk]?',
        re.IGNORECASE
    )

    for ent in entities:
        if ent.get("entity_type") == "Budget":
            raw_name = ent.get("raw_name", "")

            # Case A — no monetary signal at all → mine source text for real values
            has_monetary_signal = bool(re.search(
                r'[\$£€]|\bCC-|\b\d+(?:[.,]\d+)?[BMKbmk]\b|\bUSD\b|\bEUR\b',
                raw_name, re.IGNORECASE
            ))
            if not has_monetary_signal:
                found_values = MONEY_PATTERN.findall(state.cleaned_text)
                if found_values:
                    logger.info("[Extractor] Replacing vague Budget '%s' with mined values: %s", raw_name, found_values)
                    for val in set(found_values):   # set() deduplicates early
                        normalised.append({
                            "raw_name": val.strip(),
                            "entity_type": "Budget",
                            "attributes": {"source_phrase": raw_name},
                        })
                    continue
                normalised.append(ent)
                continue

            # Case B — has a monetary signal but LLM wrapped it with extra words
            # e.g. "$24M allocation", "$21M base amount", "$3M contingency component"
            # Strip to bare value so BUDGET_PATTERN in the resolver matches it.
            bare = BARE_MONEY.search(raw_name)
            if bare and bare.group().strip() != raw_name.strip():
                logger.info("[Extractor] Stripping Budget '%s' → '%s'", raw_name, bare.group().strip())
                ent = dict(ent)   # shallow copy — don't mutate original
                ent["attributes"] = {**ent.get("attributes", {}), "original_phrase": raw_name}
                ent["raw_name"] = bare.group().strip()

        normalised.append(ent)

    logger.info("[Extractor] Found %d entities (%d after normalisation).", len(entities), len(normalised))
    cleaned = _clean_entity_list(normalised)
    if len(cleaned) < len(normalised):
        logger.info("[Extractor] Post-clean: %d → %d entities", len(normalised), len(cleaned))
    return {"extracted_entities": cleaned}


RELATIONSHIP_SYSTEM_PROMPT = """\You are an enterprise relationship extractor for a knowledge graph.

TASK: Given artifact text and a list of known entities, extract every relationship
between those entities that is stated or clearly implied by the text.

Think step-by-step: for each sentence, identify (1) the subject, (2) the verb/action,
(3) the object, (4) any qualifier. Then form a triple.

CORE PRINCIPLES:

  1. VERB-DRIVEN: Relationships come from actions and verbs, not from co-occurrence.
     "Finance approved $8M for Epsilon" →
         Finance  APPROVED           Epsilon
         Epsilon  HAS_BUDGET         $8M
     Budget belongs to the PROJECT/INITIATIVE, not to the approving org.

  2. QUALIFIER PRESERVATION: Hedges and conditionals change the relationship TYPE.
     They must not be flattened to generic APPROVED or LINKED_TO.
     Use the qualifier AS the relationship type:
       "agreed in principle"         → AGREED_IN_PRINCIPLE
       "proposed" / "suggesting"     → PROPOSED
       "subject to X"                → CONDITIONAL_ON   (source→target: project→condition)
       "prohibited until X"          → BLOCKED_BY       (source→target: budget→condition)
       "reserved but not disbursed"  → RESERVED
       "under discussion"            → UNDER_DISCUSSION
       "confirmed pending X"         → CONFIRMED_PENDING

  3. EXCLUSION EDGES: Negative scope statements are IMPORTANT governance information.
     Always create an explicit edge for them:
       "X does not apply to Y"       → X  DOES_NOT_APPLY_TO  Y
       "X is separate from Y"        → X  DISTINCT_FROM       Y
       "X excludes Y"                → X  EXCLUDES            Y

  4. NEGATED ACTIONS: When an ACTION (not an entity) is negated, create NO edge.
       "no formal memo was issued"   → DO NOT create any memo relationship
       "disbursement is prohibited"  → DO NOT create DISBURSED; DO create BLOCKED_BY

  5. COMMUNICATION EDGES: Extract sender/recipient from greetings and sign-offs.
       "Hi Sarah" / "Dear Marco"     → sender ADDRESSED_TO Sarah/Marco
       "Thanks, David" / "— Jane"    → email/note SENT_BY David/Jane

  6. SOURCE/TARGET MATCHING: Use fuzzy name matching against the entity list.
     You MAY reference an entity from the text that was missed by extraction if it
     is clearly a named thing (add it with the name from the text).

RELATIONSHIP TYPE REFERENCE:
  APPROVED / ALLOCATED / AUTHORIZED   — explicit decisions
  AGREED_IN_PRINCIPLE                 — soft agreement, not binding
  PROPOSED / UNDER_DISCUSSION         — not yet decided
  CONFIRMED / CONFIRMED_PENDING       — verified, possibly conditional
  HAS_BUDGET / FUNDED_BY             — financial allocation
  RESERVED / DISBURSED               — funds state
  CONDITIONAL_ON                      — project/budget depends on condition
  BLOCKED_BY                          — action cannot proceed until resolved
  DOES_NOT_APPLY_TO / EXCLUDES        — scope exclusion
  DISTINCT_FROM                       — disambiguation boundary
  LEADS / MANAGES / OWNS              — ownership
  DELIVERS / INTEGRATES               — delivery
  MUST_NOTIFY / ADDRESSED_TO         — communication
  SENT_BY                             — authorship
  REJECTED / INSUFFICIENT             — negative decisions

Return ONLY valid JSON — no markdown fences:
{
  "relationships": [
    {
      "source": "<entity name>",
      "target": "<entity name>",
      "type": "<RELATIONSHIP_TYPE>",
      "evidence": "<verbatim short phrase from text>"
    }
  ]
}
"""


def node_relationship_extractor(state: NexusState, config: RunnableConfig) -> dict:
    """
    Node 1b — Relationship Extractor Agent.
    Uses LLM to extract evidence-backed (subject, predicate, object) triples
    from the cleaned text, constrained to the accepted entity list.
    Runs after the resolver so it only works with confirmed entity names.
    """
    logger.info("[RelationshipExtractor] Extracting relationships from text.")

    # Include ALL entities: accepted, quarantined, AND conflicts still pending.
    # Conflicts are entities the resolver couldn't match to master data (e.g. new
    # project names like "Phoenix Upgrade") — they still exist in the text and
    # must be available for relationship extraction.
    all_entities = (
        state.resolved_entities +
        state.quarantined_entities +
        state.extracted_entities   # conflicts still in the pipeline
    )
    if not all_entities:
        return {"extracted_relationships": []}

    entity_names = [e.get("raw_name", "") for e in all_entities if e.get("raw_name")]
    logger.info("[RelationshipExtractor] Working with entities: %s", entity_names)

    source_text = state.coref_text if state.coref_text else state.cleaned_text
    messages = [
        SystemMessage(content=RELATIONSHIP_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"ARTIFACT TEXT:\n\n{source_text}\n\n"
            f"KNOWN ENTITIES: {entity_names}\n\n"
            "Extract all relationships from the ARTIFACT TEXT above using the KNOWN ENTITIES list."
        )),
    ]

    response = LLM.invoke(messages)

    payload = _robust_parse_json(response.content, "RelationshipExtractor")
    relationships = payload.get("relationships", [])

    if not relationships:
        logger.warning(
            "[RelationshipExtractor] Got 0 relationships. Payload keys: %s",
            list(payload.keys())
        )

    logger.info("[RelationshipExtractor] Extracted %d relationships.", len(relationships))
    for r in relationships:
        logger.info("  [REL] %s -[%s]-> %s | evidence: %s",
            r.get("source","?"), r.get("type","?"), r.get("target","?"), r.get("evidence",""))
    return {"extracted_relationships": relationships}


# ---------------------------------------------------------------------------
# SELF-CRITIQUE NODE  — "What did I miss?"
# ---------------------------------------------------------------------------
# After the extractor runs, a second LLM pass re-reads the ORIGINAL text
# and the extraction output, then identifies missed entities.
# This is the key to universality: instead of prompting harder, we verify.
# ---------------------------------------------------------------------------

CRITIQUE_SYSTEM_PROMPT = """\You are a quality-control reviewer for an enterprise entity extractor.

You will be given:
  1. The original artifact text (after coreference resolution)
  2. The list of entities already extracted

Your job: identify any business entity that was MISSED — present in the text
but absent from the extracted list.

WHAT TO LOOK FOR (common extraction blind spots):
  - Names in email greetings ("Hi X") or sign-offs ("Thanks, X", "— X")
  - Project/initiative names WITHOUT trigger words (capitalised noun phrases)
  - Names of organisations acting as decision-makers or gates (even lowercase)
  - Prerequisites, clearances, or sign-off gates ("X clearance", "Y approval")
  - Entities mentioned inside NEGATED or EXCLUSION sentences —
    these are real entities even though they're being excluded
  - Budget amounts expressed informally that coref may have missed
  - A second project mentioned to contrast with the main one

DO NOT flag:
  - Entities that are already in the extracted list (even under slightly different names)
  - Negated actions or documents explicitly stated NOT to exist
    ("no formal memo was issued" → "formal memo" should NOT be added)

Return ONLY valid JSON:
{
  "missed_entities": [
    {
      "raw_name": "<exact string from text>",
      "entity_type": "<Person|Product|Project|Budget|Organization|Condition>",
      "reason": "<one sentence: why this was missed and why it should be included>"
    }
  ]
}
If nothing was missed, return: {"missed_entities": []}
"""


def node_entity_critique(state: NexusState, config: RunnableConfig) -> dict:
    """
    Node 1c — Entity Self-Critique.
    A second LLM pass that reviews extraction output against the source text
    and recovers any missed entities. Runs after initial extraction, before resolver.
    This makes the pipeline robust to novel inputs without needing prompt patches.
    """
    logger.info("[EntityCritique] Reviewing extraction for missed entities.")

    source_text = state.coref_text if state.coref_text else state.cleaned_text
    already_extracted = [e.get("raw_name", "") for e in state.extracted_entities]

    messages = [
        SystemMessage(content=CRITIQUE_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"ARTIFACT TEXT:\n{source_text}\n\n"
            f"ALREADY EXTRACTED: {already_extracted}\n\n"
            "What important entities were missed? Return JSON."
        )),
    ]

    response = LLM.invoke(messages)
    payload = _robust_parse_json(response.content, "EntityCritique")
    missed = payload.get("missed_entities", [])

    if not missed:
        logger.info("[EntityCritique] No missed entities found.")
        return {}

    logger.info("[EntityCritique] Recovered %d missed entities:", len(missed))
    for m in missed:
        logger.info("  + %-30s (%s) — %s",
            m.get("raw_name","?"), m.get("entity_type","?"), m.get("reason",""))

    # Merge missed entities into extracted_entities for the resolver
    # Strip the "reason" field — resolver doesn't need it
    recovered = [
        {"raw_name": m["raw_name"], "entity_type": m["entity_type"], "attributes": {}}
        for m in missed
        if m.get("raw_name") and m.get("entity_type")
    ]

    merged = state.extracted_entities + recovered
    cleaned = _clean_entity_list(merged)
    return {"extracted_entities": cleaned}


def _query_master_data_graph(entities: list[dict]) -> dict[str, dict]:
    """
    Query Master Data Graph for candidate matches.
    Uses exact → substring → fuzzy fallback so LLM name variants still resolve.
    Returns: { raw_name -> { canonical_id, similarity } }
    """
    import difflib

    MASTER_DATA = {
        # Projects
        "orion-v2":          {"canonical_id": "MDG-PROJ-00142", "similarity": 0.95},
        "project orion":     {"canonical_id": "MDG-PROJ-00142", "similarity": 0.93},
        "orion":             {"canonical_id": "MDG-PROJ-00142", "similarity": 0.87},
        "epsilon rollout":         {"canonical_id": "MDG-PROJ-00201", "similarity": 0.96},
        "horizon modernization":   {"canonical_id": "MDG-PROJ-00301", "similarity": 0.95},
        "horizon analytics":       {"canonical_id": "MDG-PROJ-00302", "similarity": 0.95},
        "horizon":                 {"canonical_id": "MDG-PROJ-00300", "similarity": 0.70},
        "epsilon":           {"canonical_id": "MDG-PROJ-00201", "similarity": 0.72},
        "epsilon-x":         {"canonical_id": "MDG-PROJ-00202", "similarity": 0.96},
        "phoenix upgrade":   {"canonical_id": "MDG-PROJ-00203", "similarity": 0.94},
        "phoenix":           {"canonical_id": "MDG-PROJ-00203", "similarity": 0.75},
        # ── FIX 1: Added Helix family, governance orgs, account codes,
        #           Q-series dashboards, and common budget amounts so the
        #           fuzzy matcher can't incorrectly cross-match them.
        "helix transformation":  {"canonical_id": "MDG-PROJ-00401", "similarity": 0.91},
        "helix analytics":       {"canonical_id": "MDG-PROJ-00402", "similarity": 0.91},
        "helix":                 {"canonical_id": "MDG-PROJ-00400", "similarity": 0.72},
        # Products / Tech
        "sap s/4hana":    {"canonical_id": "MDG-PROD-00001", "similarity": 0.99},
        "sap btp":        {"canonical_id": "MDG-PROD-00010", "similarity": 0.99},
        "hana cloud":     {"canonical_id": "MDG-PROD-00011", "similarity": 0.97},
        "q1 dashboard":   {"canonical_id": "MDG-PROD-00221", "similarity": 0.84},
        "q2 dashboard":   {"canonical_id": "MDG-PROD-00222", "similarity": 0.84},
        "q3 dashboard":   {"canonical_id": "MDG-PROD-00223", "similarity": 0.84},
        "q4 dashboard":   {"canonical_id": "MDG-PROD-00224", "similarity": 0.84},
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
        "legal":          {"canonical_id": "MDG-ORG-LEGAL",  "similarity": 0.78},
        "compliance":     {"canonical_id": "MDG-ORG-COMPL",  "similarity": 0.78},
        "treasury":       {"canonical_id": "MDG-ORG-TREAS",  "similarity": 0.78},
        "procurement":    {"canonical_id": "MDG-ORG-PROC",   "similarity": 0.78},
        "audit":          {"canonical_id": "MDG-ORG-AUDIT",  "similarity": 0.78},
        "board":          {"canonical_id": "MDG-ORG-BOARD",  "similarity": 0.78},
        "sap se":         {"canonical_id": "MDG-ORG-00001",  "similarity": 0.99},
        "titancorp":      {"canonical_id": "MDG-ORG-00318",  "similarity": 0.90},
        "cloudbase inc":  {"canonical_id": "MDG-ORG-00401",  "similarity": 0.88},
        # Account classification codes
        "opex":           {"canonical_id": "MDG-ACC-OPEX",   "similarity": 0.88},
        "capex":          {"canonical_id": "MDG-ACC-CAPEX",  "similarity": 0.88},
        # Budgets / Cost Centres
        "cc-7740":        {"canonical_id": "MDG-CC-7740",    "similarity": 0.99},
        "cc-9981":        {"canonical_id": "MDG-CC-9981",    "similarity": 0.99},
        "cc-5510":        {"canonical_id": "MDG-CC-5510",    "similarity": 0.99},
        "cc-8821":        {"canonical_id": "MDG-CC-8821",    "similarity": 0.99},
        "$3m":            {"canonical_id": "MDG-BUDG-0300",  "similarity": 0.82},
        "$5m":            {"canonical_id": "MDG-BUDG-0500",  "similarity": 0.82},
        "$8m":            {"canonical_id": "MDG-BUDG-0800",  "similarity": 0.82},
        "$10m":           {"canonical_id": "MDG-BUDG-1000",  "similarity": 0.82},
        "$15m":           {"canonical_id": "MDG-BUDG-1500",  "similarity": 0.82},
        "$20m":           {"canonical_id": "MDG-BUDG-2000",  "similarity": 0.82},
        "$21m":           {"canonical_id": "MDG-BUDG-2100",  "similarity": 0.82},
        "$24m":           {"canonical_id": "MDG-BUDG-2400",  "similarity": 0.82},
        "$25m":           {"canonical_id": "MDG-BUDG-2500",  "similarity": 0.82},
        "$28m":           {"canonical_id": "MDG-BUDG-2800",  "similarity": 0.82},
        "$30m":           {"canonical_id": "MDG-BUDG-3000",  "similarity": 0.82},
        "$38m":           {"canonical_id": "MDG-BUDG-0831",  "similarity": 0.82},
        "$40m":           {"canonical_id": "MDG-BUDG-4000",  "similarity": 0.82},
        "$42m":           {"canonical_id": "MDG-BUDG-0842",  "similarity": 0.82},
        "$50m":           {"canonical_id": "MDG-BUDG-0842",  "similarity": 0.82},
        "$3.2m":          {"canonical_id": "MDG-BUDG-0320",  "similarity": 0.82},
        "$4m":            {"canonical_id": "MDG-BUDG-0804",  "similarity": 0.82},
        "$4.2b":          {"canonical_id": "MDG-BUDG-0420",  "similarity": 0.80},
        "usd 1.2m":       {"canonical_id": "MDG-BUDG-0120",  "similarity": 0.84},
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

    # ── FIX 3: Raised fallback scores so novel-but-real entities don't
    #    loop forever. Project 0.50→0.72, Org 0.48→0.75, Budget 0.72→0.82.
    TYPE_FALLBACK_SCORE = {
        "Budget":       0.82,   # monetary values are fully self-identifying
        "Person":       0.55,   # single names are ambiguous → conflict tier
        "Project":      0.72,   # named initiatives are self-identifying in governance docs
        "Product":      0.55,   # unknown products need review but not permanent conflict
        "Organization": 0.75,   # governance actors (Finance, Legal, Board) are self-identifying
        "Organisation": 0.75,
        "Condition":    0.72,   # clearances / gates are self-describing in context
        "Unknown":      0.20,
    }

    # ── FIX 4: Pre-resolve deduplication ─────────────────────────────────────
    # The LLM (across coref + extract + critique passes) may surface the same
    # entity from multiple text mentions. Collapse duplicates NOW — before
    # scoring — so a value like $24M that appears 4 times enters the loop once.
    # Budget key = bare monetary value; all other types = normalised raw_name.
    _BARE_KEY_RX = re.compile(
        r'[\$£€]\d+(?:[.,]\d+)?[BMKbmk]?'
        r'|\bCC-\d{3,6}\b'
        r'|\bUSD\s*\d+(?:[.,]\d+)?[BMKbmk]?',
        re.IGNORECASE
    )

    def _entity_dedup_key(e: dict) -> str:
        raw = e.get("raw_name", "").strip()
        if e.get("entity_type") in ("Budget", "budget"):
            m = _BARE_KEY_RX.search(raw)
            if m:
                return m.group().lower()
        return raw.lower()

    _pre_seen: set = set()
    _deduped_input: list = []
    for _e in state.extracted_entities:
        _k = _entity_dedup_key(_e)
        if _k and _k not in _pre_seen:
            _pre_seen.add(_k)
            _deduped_input.append(_e)

    if len(_deduped_input) < len(state.extracted_entities):
        logger.info(
            "[Resolver] Pre-dedup: %d → %d entities (%d duplicates collapsed)",
            len(state.extracted_entities), len(_deduped_input),
            len(state.extracted_entities) - len(_deduped_input),
        )

    master_data = _query_master_data_graph(_deduped_input)

    accepted, conflicts, quarantined = [], [], []

    for ent in _deduped_input:
        candidate = master_data.get(ent["raw_name"], {})
        similarity = float(candidate.get("similarity", 0.0))
        canonical_id = candidate.get("canonical_id")

        # If no MDG match, use entity-type fallback or dynamic canonicalisation
        if similarity == 0.0:
            entity_type = ent.get("entity_type", ent.get("type", "Unknown"))
            raw = ent.get("raw_name", "")

            # ── Dynamic Budget canonicalisation ──────────────────────────────
            # Any valid monetary value is self-identifying — it doesn't need
            # a hardcoded master data entry. Generate a canonical ID from the
            # value itself and accept it at 0.82 (same as hardcoded budgets).
            BUDGET_PATTERN = re.compile(
                r'^[\$£€]\d+(?:[.,]\d+)?[BMKbmk]?$'
                r'|^\d+(?:[.,]\d+)?[BMKbmk]$'
                r'|^USD\s+\d+(?:[.,]\d+)?[BMKbmk]?$'
                r'|^CC-\d{3,6}$',
                re.IGNORECASE
            )
            if entity_type in ("Budget", "budget") and BUDGET_PATTERN.match(raw.strip()):
                slug = re.sub(r'[^a-zA-Z0-9]', '-', raw.upper()).strip('-')
                canonical_id = f"MDG-BUDG-{slug}"
                similarity = 0.82
            # ── Dynamic Organization canonicalisation ─────────────────────────
            # Common single-word org names (Finance, Legal, Compliance, etc.)
            # are self-identifying governance actors — accept at 0.75.
            elif entity_type in ("Organization", "Organisation") and len(raw.split()) <= 3:
                slug = re.sub(r'[^a-zA-Z0-9]', '-', raw.upper()).strip('-')
                canonical_id = f"MDG-ORG-{slug}"
                similarity = 0.75
            # ── Dynamic Condition canonicalisation ────────────────────────────
            elif entity_type == "Condition":
                slug = re.sub(r'[^a-zA-Z0-9]', '-', raw.upper()).strip('-')
                canonical_id = f"MDG-COND-{slug}"
                similarity = 0.72   # above accept threshold
            else:
                similarity = TYPE_FALLBACK_SCORE.get(entity_type, 0.20)
                canonical_id = None  # genuinely unknown — needs human review

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
        / max(len(_deduped_input), 1)
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

    # ── FIX 5: Dedup all three buckets — not just accepted ───────────────────
    # Prevents the same conflict entity re-entering the loop on the next
    # iteration and accumulating duplicate rows.
    def _dedup(entity_list: list, exclude_keys: set | None = None) -> tuple:
        """Return (deduplicated_list, keys_used). Budget-aware keying."""
        excl = set(exclude_keys or [])
        out: list = []
        used: set = set()
        for e in entity_list:
            k = e.get("canonical_id") or _entity_dedup_key(e)
            if k and k not in used and k not in excl:
                used.add(k)
                out.append(e)
        return out, used

    # Keys already committed in prior resolver loops
    prior_keys = {
        e.get("canonical_id") or _entity_dedup_key(e)
        for e in state.resolved_entities
    }

    deduped_accepted,    acc_keys  = _dedup(accepted,    prior_keys)
    deduped_conflicts,   conf_keys = _dedup(conflicts,   prior_keys | acc_keys)
    deduped_quarantined, _         = _dedup(quarantined, prior_keys | acc_keys | conf_keys)

    conflict_fingerprint = ",".join(sorted(e.get("raw_name", "") for e in deduped_conflicts))

    return {
        "resolved_entities":    state.resolved_entities + deduped_accepted,
        "extracted_entities":   deduped_conflicts,          # only unique conflicts loop back
        "quarantined_entities": state.quarantined_entities + deduped_quarantined,
        "resolver_loop_count":  state.resolver_loop_count + 1,
        "last_conflict_fingerprint": conflict_fingerprint,
    }


def node_graph_architect(state: NexusState, config: RunnableConfig) -> dict:
    """
    Node 3 — Graph Architect Agent.
    Writes reconciled entity triples into Neo4j Knowledge Graph.
    """
    logger.info("[GraphArchitect] Writing %d entities to Knowledge Graph.", len(state.resolved_entities))

    # --- Neo4j Write — reads from environment (set in docker-compose.yml) ---
    import os
    NEO4J_URI  = os.getenv("NEO4J_URI",      "bolt://neo4j:7687")
    NEO4J_USER = os.getenv("NEO4J_USER",     "neo4j")
    NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "password")
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
    Consensus Logic Router.
    - Still have conflicts AND loops remaining → re-run resolver
    - Max loops hit OR no conflicts → proceed to Graph Architect
    - All entities quarantined → end with quarantine warning
    """
    has_conflicts = len(state.extracted_entities) > 0
    loop_exhausted = state.resolver_loop_count >= state.max_resolver_loops
    has_accepted = len(state.resolved_entities) > 0

    # Check if conflicts are stale — same entities failing every loop means
    # re-running the resolver will never help (not in master data)
    current_fingerprint = ",".join(sorted(e.get("raw_name","") for e in state.extracted_entities))
    conflicts_are_stale = (current_fingerprint == state.last_conflict_fingerprint and
                           state.resolver_loop_count >= 1)

    if has_conflicts and not loop_exhausted and not conflicts_are_stale:
        logger.info("[Router] Conflicts remain. Looping back to resolver (loop %d).", state.resolver_loop_count)
        return "resolver"

    if conflicts_are_stale:
        logger.info("[Router] Conflicts unchanged since last loop — no point re-running. Proceeding.")

    # Loop exhausted or no conflicts — proceed if we have ANY accepted entities
    # (accumulated across all loops, not just this one)
    if has_accepted:
        logger.info("[Router] Routing to Graph Architect with %d accepted entities.", len(state.resolved_entities))
        return "graph_architect"

    logger.warning("[Router] No accepted entities across all loops. Ending pipeline.")
    return END


# ---------------------------------------------------------------------------
# 5. LANGGRAPH STATE MACHINE ASSEMBLY
# ---------------------------------------------------------------------------

def build_nexus_graph() -> StateGraph:
    """
    Assembles the BTP-Nexus LangGraph state machine.

    Flow:
        START → denoise → coreference → extractor → entity_critique
             → resolver ⟲ (up to N loops)
                    └─ relationship_extractor → graph_architect → END
                    └─ END (all quarantined)
    """
    graph = StateGraph(NexusState)

    # Register nodes
    graph.add_node("denoise", node_denoise)
    graph.add_node("coreference", node_coreference)
    graph.add_node("extractor", node_extractor)
    graph.add_node("entity_critique", node_entity_critique)
    graph.add_node("resolver", node_resolver)
    graph.add_node("relationship_extractor", node_relationship_extractor)
    graph.add_node("graph_architect", node_graph_architect)

    # Define edges
    graph.add_edge(START, "denoise")
    graph.add_edge("denoise", "coreference")
    graph.add_edge("coreference", "extractor")
    graph.add_edge("extractor", "entity_critique")   # self-critique recovers missed entities
    graph.add_edge("entity_critique", "resolver")

    # Conditional loop-back from resolver
    graph.add_conditional_edges(
        "resolver",
        route_resolver,
        {
            "resolver": "resolver",
            "graph_architect": "relationship_extractor",  # relationship extraction before graph write
            END: END,
        },
    )
    graph.add_edge("relationship_extractor", "graph_architect")
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
            "version": "2.0.0",
            "artifact_id": artifact_id or "unknown",
        })
        mlflow.log_param("max_resolver_loops", 3)
        mlflow.log_param("pipeline_version", "2.0.0-coref")
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
        mlflow.log_metric("final_relationships", len(final_state.extracted_relationships))
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
