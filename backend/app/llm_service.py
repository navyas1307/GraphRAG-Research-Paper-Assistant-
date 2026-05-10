"""
llm_service.py — FIXED v2 (GraphRAG LLM control + quota conservation)

Key fixes vs v1
───────────────
1. compare_papers_deep() now accepts graph_context (GraphContext | dict) and
   injects it as PRIORITY 1 input — LLM must reference graph_reason.
2. Strict prompt rules:
     - If no shared entities → MUST say "weakly related"
     - graph_reason field REQUIRED in output
     - Summaries are PRIORITY 3 (fallback only)
3. extract_all_metadata() normalizes entities with normalize_entity_list()
   immediately after parsing, so stored entities are canonical from day 1.
4. _fallback_extraction() also normalizes its keyword-matched entities.
5. Ollama / Gemini routing unchanged.
"""

import google.generativeai as genai
import json
import time
import re
import hashlib
from typing import Dict, List, Any, Optional
import os
import requests as _http
from dotenv import load_dotenv

load_dotenv()

GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
]

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.0-flash-lite")


OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
_OLLAMA_AVAILABLE: Optional[bool] = None


def _ollama_is_alive() -> bool:
    global _OLLAMA_AVAILABLE
    if _OLLAMA_AVAILABLE is not None:
        return _OLLAMA_AVAILABLE
    try:
        r = _http.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        _OLLAMA_AVAILABLE = r.status_code == 200
    except Exception:
        _OLLAMA_AVAILABLE = False
    return _OLLAMA_AVAILABLE


def _call_ollama(prompt: str) -> Optional[str]:
    if not _ollama_is_alive():
        return None
    try:
        r = _http.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip() or None
    except Exception as e:
        print(f"⚠️ Ollama call failed: {e}")
    return None


_request_timestamps: List[float] = []
MAX_REQUESTS_PER_MINUTE = 12
_metadata_cache: Dict[str, Any] = {}



def _check_rate_limit():
    global _request_timestamps
    now = time.time()
    _request_timestamps = [t for t in _request_timestamps if now - t < 60]
    if len(_request_timestamps) >= MAX_REQUESTS_PER_MINUTE:
        wait_time = 60 - (now - _request_timestamps[0]) + 1
        print(f"⏳ Rate limit: waiting {wait_time:.1f}s before next Gemini call")
        time.sleep(wait_time)
    _request_timestamps.append(time.time())


def _is_daily_quota_error(err_str: str) -> bool:
    return (
        "generativelanguage.googleapis.com/generate_content_free_tier_requests" in err_str
        or "RequestsPerDay" in err_str
        or "per_day" in err_str.lower()
        or "free_tier_requests" in err_str
    )


def _call_gemini(prompt: str, model: Optional[str] = None) -> Optional[str]:
    models_to_try = [model or GEMINI_MODEL]

    if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL not in models_to_try:
        models_to_try.append(GEMINI_FALLBACK_MODEL)

    last_error = None

    # 🔑 LOOP THROUGH KEYS
    for key_index, key in enumerate(GEMINI_KEYS):
        if not key:
            continue

        try:
            print(f"🔑 Trying Gemini key {key_index + 1}")
            genai.configure(api_key=key)

            for model_name in models_to_try:
                result = _try_gemini_model(prompt, model_name)
                if result:
                    print(f"✅ Success with key {key_index + 1} ({model_name})")
                    return result

        except Exception as e:
            print(f"❌ Key {key_index + 1} failed: {e}")
            last_error = e
            continue

    print("❌ All Gemini keys failed — trying Ollama fallback")
    return _call_ollama(prompt)


def _try_gemini_model(prompt: str, model_name: str, max_retries: int = 2) -> Optional[str]:

    gemini_model = genai.GenerativeModel(model_name)
    backoff = [5, 15]

    for attempt in range(max_retries + 1):
        try:
            _check_rate_limit()
            response = gemini_model.generate_content(prompt)
            return response.text if response and response.text else None

        except Exception as e:
            err_str   = str(e)
            err_lower = err_str.lower()

            if any(x in err_lower for x in ["api_key", "invalid", "unauthorized", "403"]):
                print(f"❌ Gemini auth error on {model_name}: {e}")
                return None

            if _is_daily_quota_error(err_str):
                print(f"⚠️ Quota exhausted for this key on {model_name}")
                return None

            if any(x in err_lower for x in ["quota", "429", "rate", "resource_exhausted"]):
                if attempt < max_retries:
                    wait = backoff[min(attempt, len(backoff) - 1)]
                    print(f"⚠️ Rate limit on {model_name} (attempt {attempt+1}), waiting {wait}s")
                    time.sleep(wait)
                    continue
                return None

            print(f"❌ Gemini error on {model_name} (attempt {attempt+1}): {type(e).__name__}: {e}")
            if attempt < max_retries:
                time.sleep(backoff[min(attempt, len(backoff) - 1)])
            else:
                return None

    return None


# ─── Entity Extraction ────────────────────────────────────────────────────────

_ENTITY_EXTRACTION_SYSTEM = """You are a research paper analyst.
Analyze the paper and return ONLY valid JSON with entities AND a structured research summary.
No markdown, no preamble, no explanation — just the JSON object."""

_ENTITY_EXTRACTION_TEMPLATE = """Analyze this research paper and return a single JSON object.

TITLE: {title}

ABSTRACT: {abstract}

{sections_block}

Return EXACTLY this JSON schema (no extra keys, no markdown):
{{
  "methods": [],
  "models": [],
  "datasets": [],
  "tasks": [],
  "topics": [],
  "research_summary": {{
    "problem_definition": "",
    "background_and_prior_work": "",
    "methodology": "",
    "technical_novelty": "",
    "experimental_setup": "",
    "results_and_analysis": "",
    "limitations": "",
    "conclusion": ""
  }},
  "research_gap": {{
    "gap_identification": "",
    "remaining_limitations": "",
    "methodological_weaknesses": "",
    "future_directions": []
  }}
}}

ENTITY RULES:
1. Each entity list: at most 5 items. Prefer precision over recall.
2. Normalize to canonical short forms: "RAG", "LLM", "fine-tuning", "NLP", "question answering".
3. No duplicates. SHORT forms only — not full sentences.

RESEARCH SUMMARY RULES (all fields required, write 2-4 sentences each):
- problem_definition: What problem does this paper solve and why does it matter?
- background_and_prior_work: What prior approaches existed and what were their limitations?
- methodology: How does the proposed system/method work technically? Be specific.
- technical_novelty: What is genuinely new compared to prior work?
- experimental_setup: What datasets, baselines, and evaluation metrics were used?
- results_and_analysis: What were the key quantitative results? Include numbers.
- limitations: What are the paper's acknowledged weaknesses or scope restrictions?
- conclusion: What is the paper's main takeaway and impact?

RESEARCH GAP RULES:
- gap_identification: What important open problem does this paper leave unsolved?
- remaining_limitations: What constraints still apply after this work?
- methodological_weaknesses: What flaws exist in the evaluation or approach?
- future_directions: List 2-4 concrete follow-on research directions as strings."""



# ─── Section builder — packs full paper into one prompt efficiently ────────────

# Priority order: intro/background first for context, then methods, experiments,
# results, conclusion. Each section is trimmed to fit a total budget so we
# maximise coverage in a single Gemini call (~12k chars ≈ ~3k tokens of body text,
# well within gemini-2.5-flash's 1M token window and cheap).
# Priority rank → (weight, cap_chars)
# Rank 0 = highest priority. Weights drive proportional budget allocation.
# Caps prevent any single section from monopolising the budget.
_SECTION_TIERS: Dict[int, tuple] = {
    0: (4.0, 10_000),  # system / architecture — most important for summary
    1: (4.0, 10_000),  # methodology / approach / model / proposed / framework
    2: (3.0, 8_000),   # experiments / evaluation / setup / implementation
    3: (3.0, 8_000),   # results / analysis / performance / ablation
    4: (1.5, 4_000),   # introduction / motivation / overview
    5: (1.0, 3_000),   # related work / background / prior
    6: (1.5, 4_000),   # conclusion / discussion / limitations / future
    7: (0.5,   500),   # unknown / appendix / acknowledgements / references
}

_SECTION_KEYWORDS: List[tuple] = [
    # (rank, keywords) — first match wins, ordered most-specific first
    (0, ["system", "architecture", "textrunner", "design", "component"]),
    (1, ["method", "approach", "model", "framework", "proposed", "our method",
         "algorithm", "technique", "formulation", "overview of"]),
    (2, ["experiment", "evaluation", "benchmark", "setup", "implementation",
         "training", "dataset", "corpus", "configuration"]),
    (3, ["result", "performance", "analysis", "ablation", "comparison",
         "accuracy", "precision", "recall", "f1", "error rate"]),
    (4, ["introduction", "intro", "motivation", "problem statement", "overview"]),
    (5, ["related", "prior", "previous", "literature", "background",
         "existing", "survey", "comparison with"]),
    (6, ["conclusion", "discussion", "future", "limitation", "summary",
         "contribution", "impact"]),
]

_BODY_BUDGET = 40_000   # total chars for sections block (~10k tokens, trivial for gemini-2.5-flash 1M ctx)


def _build_sections_block(
    sections: Optional[List[Dict]],
    full_text: Optional[str],
    abstract: str,
) -> str:
    """
    Pack the full paper into one prompt block efficiently.

    Algorithm (single API call):
      1. Rank every section by content type using _SECTION_KEYWORDS.
      2. Each tier has a pre-set weight and per-section char cap (_SECTION_TIERS).
      3. Budget = weight / total_weight * _BODY_BUDGET, capped at tier cap.
      4. Sections in the same tier are merged so subsections aren't double-penalised.
      5. Sections are emitted in *reading order* (original index), not priority order,
         so the LLM sees the paper naturally — priorities only drive char allocation.
      6. Falls back to raw full_text[:_BODY_BUDGET] if no structured sections.
    """
    if not sections:
        if full_text:
            body = full_text[len(abstract):].strip() if abstract and full_text.startswith(abstract[:80]) else full_text
            return f"PAPER BODY:\n{body[:_BODY_BUDGET]}"
        return ""

    def _rank(sec: Dict) -> int:
        title_lower = (sec.get("title") or "").lower()
        content_lower = (sec.get("content") or "")[:200].lower()
        combined = title_lower + " " + content_lower
        for rank, keywords in _SECTION_KEYWORDS:
            if any(kw in combined for kw in keywords):
                return rank
        return 7  # unknown

    # Annotate each section with its rank and original index (preserve reading order)
    annotated = [(i, _rank(s), s) for i, s in enumerate(sections)]

    # Compute total weight for proportional budget
    total_weight = sum(_SECTION_TIERS[rank][0] for _, rank, _ in annotated) or 1.0

    # Pre-compute char allocation per section
    allocs: Dict[int, int] = {}
    for i, rank, _ in annotated:
        weight, cap = _SECTION_TIERS[rank]
        alloc = int((weight / total_weight) * _BODY_BUDGET)
        allocs[i] = min(alloc, cap)

    # Emit in original reading order so the LLM sees a natural document flow
    parts: List[str] = []
    used = 0
    for i, rank, sec in annotated:
        alloc = allocs[i]
        if alloc < 80:
            continue  # skip refs / ack with negligible budget

        # Gather content: direct + subsections
        direct = (sec.get("content") or "").strip()
        subs   = sec.get("subsections") or []
        sub_text = " ".join((s.get("content") or "") for s in subs).strip()
        full_sec = (direct + " " + sub_text).strip()

        if not full_sec:
            continue

        snippet = full_sec[:alloc]
        parts.append(f"[{sec.get('title', 'Section')}]\n{snippet}")
        used += len(snippet)

    if not parts:
        return f"PAPER BODY:\n{(full_text or '')[:_BODY_BUDGET]}"

    return "PAPER SECTIONS (reading order):\n\n" + "\n\n".join(parts)


def extract_all_metadata(
    title: str,
    abstract: str,
    sections: Optional[List[Dict]] = None,
    full_text: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Extract structured metadata via LLM.
    FIXED: normalize_entity_list() is applied to ALL entity fields immediately
    after parsing, so stored entities are canonical from the first write.
    """
    from app.entity_normalizer import normalize_entity_list, normalize_paper_entities

    cache_key = hashlib.md5(f"{title}{(abstract or '')[:200]}".encode()).hexdigest()
    if not force and cache_key in _metadata_cache:
        return _metadata_cache[cache_key]

    sections_block = _build_sections_block(sections, full_text, abstract)

    prompt = _ENTITY_EXTRACTION_TEMPLATE.format(
        title=title,
        abstract=(abstract or "")[:1500],
        sections_block=sections_block,
    )

    response_text = _call_gemini(prompt)

    if response_text:
        try:
            clean  = re.sub(r"```(?:json)?\s*|\s*```", "", response_text).strip()
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except Exception:
                    parsed = {}
            else:
                parsed = {}

        if isinstance(parsed, dict):
            # Ensure all entity list keys exist
            for fld in ("methods", "models", "datasets", "tasks", "topics"):
                if fld not in parsed or not isinstance(parsed[fld], list):
                    parsed[fld] = []

            # Ensure research_summary has all required string fields
            rs_llm = parsed.get("research_summary") or {}
            if not isinstance(rs_llm, dict):
                rs_llm = {}
            rs_fields = (
                "problem_definition", "background_and_prior_work", "methodology",
                "technical_novelty", "experimental_setup", "results_and_analysis",
                "limitations", "conclusion",
            )
            for f in rs_fields:
                if not isinstance(rs_llm.get(f), str):
                    rs_llm[f] = ""
            # If LLM returned nothing useful, seed problem_definition from abstract
            if not any(rs_llm.get(f, "").strip() for f in rs_fields):
                rs_llm["problem_definition"] = abstract if abstract else title
            parsed["research_summary"] = rs_llm

            # Ensure research_gap has all required fields
            rg_llm = parsed.get("research_gap") or {}
            if not isinstance(rg_llm, dict):
                rg_llm = {}
            for f in ("gap_identification", "remaining_limitations", "methodological_weaknesses"):
                if not isinstance(rg_llm.get(f), str):
                    rg_llm[f] = ""
            if not isinstance(rg_llm.get("future_directions"), list):
                rg_llm["future_directions"] = []
            parsed["research_gap"] = rg_llm

            # Normalize entities
            entities_raw = {
                k: parsed.get(k, []) for k in ("methods", "models", "datasets", "tasks", "topics")
            }
            fake_paper = {"paper_id": cache_key, "entities": entities_raw, "topics": parsed.get("topics", [])}
            normalize_paper_entities(fake_paper)
            parsed.update({k: fake_paper["entities"][k] for k in ("methods", "models", "datasets", "tasks", "topics")})
            parsed["topics"] = fake_paper["topics"]

            result = _build_metadata_result(parsed, title, abstract)
            _metadata_cache[cache_key] = result
            return result

    result = _fallback_extraction(title, abstract)
    _metadata_cache[cache_key] = result
    return result


def _build_metadata_result(parsed: dict, title: str, abstract: str) -> dict:
    """Assemble final metadata dict from parsed LLM output."""
    # research_summary and research_gap are already validated by the caller;
    # trust what the LLM returned — only fall back if completely empty.
    rs = parsed.get("research_summary") or {}
    rg = parsed.get("research_gap")     or {}

    # Fallback only if every field is blank (e.g. LLM quota exhausted path)
    has_any_rs = any(isinstance(v, str) and v.strip() for v in rs.values())
    if not has_any_rs:
        rs = {
            "problem_definition":        abstract if abstract else title,
            "background_and_prior_work": "",
            "methodology":               ", ".join(parsed.get("methods", [])[:5]) or "",
            "technical_novelty":         "",
            "experimental_setup":        ", ".join(parsed.get("datasets", [])[:3]) or "",
            "results_and_analysis":      "",
            "limitations":               "",
            "conclusion":                "",
        }

    return {
        "research_summary":  rs,
        "research_gap":      rg or {
            "gap_identification": "", "remaining_limitations": "",
            "methodological_weaknesses": "", "future_directions": [],
        },
        "entities": {
            "methods":  parsed.get("methods",  []),
            "models":   parsed.get("models",   []),
            "datasets": parsed.get("datasets", []),
            "tasks":    parsed.get("tasks",    []),
            "topics":   parsed.get("topics",   []),
        },
        "topics":            parsed.get("topics", []),
        "section_summaries": parsed.get("section_summaries", {}),
    }


# ─── FIXED: GraphRAG-aware multi-paper comparison ─────────────────────────────

def compare_papers_deep(
    papers_data: List[Dict[str, Any]],
    graph_context=None,   # GraphContext dataclass OR dict OR None
) -> Dict[str, Any]:
    """
    PhD-level structured comparison.

    FIXED: graph_context is injected as PRIORITY 1.
    Strict LLM rules:
      - If no shared entities → say "weakly related" in problem_solution_relationship
      - graph_reason field REQUIRED
      - Summaries used only as supplementary context (PRIORITY 3)
    """
    if len(papers_data) < 2:
        return _empty_comparison()

    # ── Build graph context string ─────────────────────────────────────────────
    graph_ctx_str = ""
    signal_type   = "none"

    if graph_context is not None:
        try:
            from app.graph_context import graph_context_to_str, GraphContext
            if isinstance(graph_context, GraphContext):
                graph_ctx_str = graph_context_to_str(graph_context)
                signal_type   = graph_context.signal_type
            elif isinstance(graph_context, dict):
                # Build a minimal GraphContext from dict
                gc = GraphContext(
                    shared_methods  = graph_context.get("shared_methods", []),
                    shared_datasets = graph_context.get("shared_datasets", []),
                    shared_topics   = graph_context.get("shared_topics", []),
                    has_signal      = graph_context.get("has_signal", False),
                    signal_type     = graph_context.get("signal_type", "none"),
                )
                graph_ctx_str = graph_context_to_str(gc)
                signal_type   = gc.signal_type
        except Exception as e:
            print(f"⚠️ Could not serialize graph_context: {e}")

    # ── Build paper blocks (PRIORITY 3 — summaries as fallback only) ──────────
    paper_blocks = []
    for i, p in enumerate(papers_data, 1):
        rs  = p.get("research_summary") or {}
        ent = p.get("entities") or {}

        # Use normalized entities if available
        from app.entity_normalizer import normalize_entity_list
        methods  = normalize_entity_list(ent.get("methods",  []) or [])[:5]
        datasets = normalize_entity_list(ent.get("datasets", []) or [])[:4]
        topics   = normalize_entity_list(p.get("topics",     []) or [])[:5]

        block = (
            f"PAPER {i}: {p.get('title', 'Unknown')}\n"
            f"Methods: {', '.join(methods) or 'N/A'}\n"
            f"Datasets: {', '.join(datasets) or 'N/A'}\n"
            f"Topics: {', '.join(topics) or 'N/A'}\n"
            f"Problem (summary): {rs.get('problem_definition', 'N/A')[:600]}\n"
            f"Methodology (summary): {rs.get('methodology', 'N/A')[:400]}\n"
            f"Results (summary): {rs.get('results_and_analysis', 'N/A')[:400]}"
        )
        paper_blocks.append(block)

    combined = "\n\n".join(paper_blocks)
    n        = len(papers_data)

    sw_template = "\n".join([
        f'"paper{i}": {{"strengths": ["s1", "s2"], "weaknesses": ["w1", "w2"]}}'
        for i in range(1, n + 1)
    ])
    uc_template = "\n".join([
        f'"paper{i}": "use case"' for i in range(1, n + 1)
    ])

    # ── Strict GraphRAG prompt ─────────────────────────────────────────────────
    no_entity_instruction = (
        '  - problem_solution_relationship MUST include the phrase "weakly related" or "no shared entities".\n'
        if signal_type == "none" else ""
    )

    graph_section = (
        f"\n{graph_ctx_str}\n\n"
        "GRAPH RULES (STRICT — higher priority than summaries):\n"
        "  - PRIORITY 1: Use graph_context above as ground truth for entity overlap.\n"
        "  - PRIORITY 2: Use Methods/Datasets/Topics entity lists.\n"
        "  - PRIORITY 3: Use problem/methodology/results summaries as supplementary only.\n"
        "  - graph_reason field MUST be populated (copy from graph_context above).\n"
        f"{no_entity_instruction}"
        "  - Do NOT infer similarity from text alone when graph says no overlap.\n"
    ) if graph_ctx_str else ""

    prompt = f"""You are an expert AI research analyst. Compare these {n} papers for a PhD-level audience.
Return ONLY valid JSON (no markdown, no backticks).
{graph_section}
{combined}

Return EXACTLY this JSON:
{{
  "comparison_table": {{
    "problem":     [{', '.join([f'"paper{i} problem in 1 sentence"' for i in range(1, n+1)])}],
    "methodology": [{', '.join([f'"paper{i} core method in 1 sentence"' for i in range(1, n+1)])}],
    "datasets":    [{', '.join([f'"paper{i} datasets"' for i in range(1, n+1)])}],
    "results":     [{', '.join([f'"paper{i} key result with numbers"' for i in range(1, n+1)])}]
  }},
  "key_differences": ["difference 1", "difference 2", "difference 3"],
  "problem_solution_relationship": "3-4 sentences referencing graph signal strength.",
  "strengths_vs_weaknesses": {{
    {sw_template}
  }},
  "best_use_cases": {{
    {uc_template}
  }},
  "research_evolution": "3-4 sentences on how these papers advance the field.",
  "graph_reason": "copy graph_reason from GRAPH CONTEXT above, or 'no shared entities' if none"
}}"""

    response_text = _call_gemini(prompt)

    if response_text:
        try:
            clean  = re.sub(r"```(?:json)?\s*|\s*```", "", response_text).strip()
            result = json.loads(clean)
            return _validate_comparison(result, n)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                    return _validate_comparison(result, n)
                except Exception:
                    pass

    return _empty_comparison()


def _validate_comparison(result: dict, n: int) -> dict:
    defaults = _empty_comparison()
    for key in defaults:
        if key not in result:
            result[key] = defaults[key]
    # Ensure graph_reason is always present
    if "graph_reason" not in result:
        result["graph_reason"] = "not provided"
    return result


def _empty_comparison() -> dict:
    return {
        "comparison_table":              {"problem": [], "methodology": [], "datasets": [], "results": []},
        "key_differences":               [],
        "problem_solution_relationship": "",
        "strengths_vs_weaknesses":       {},
        "best_use_cases":                {},
        "research_evolution":            "",
        "graph_reason":                  "no shared entities",
    }


# ─── Legacy narrative compare ─────────────────────────────────────────────────

def compare_papers_narrative(papers_data: List[Dict[str, Any]]) -> str:
    from app.entity_normalizer import normalize_entity_list

    summaries = "\n\n".join([
        f"Paper {i+1}: {p['title']}\n"
        f"Methods: {', '.join(normalize_entity_list(p.get('entities', {}).get('methods', [])))}\n"
        f"Topics: {', '.join(normalize_entity_list(p.get('topics', [])))}"
        for i, p in enumerate(papers_data)
    ])

    prompt = f"""Compare these research papers for a researcher audience (4-5 sentences):

{summaries}

Cover: (1) shared research goal, (2) how approaches differ technically,
(3) which performs better and by how much, (4) complementary strengths."""

    response = _call_gemini(prompt)
    return response.strip() if response else "Comparison based on structural analysis below."


# ─── Cluster AI summary ────────────────────────────────────────────────────────

def summarize_cluster(
    papers_in_cluster: List[Dict[str, Any]],
    cluster_label: str,
    signal_type: str = "none",
) -> str:
    """
    Generate an AI explanation of WHY these papers were grouped into the same
    cluster. Designed for the graph page cluster-info panel.

    Three bugs this fixes vs using compare_papers_narrative() for clusters:

    BUG 1 — Missing papers from summary
      compare_papers_narrative() builds context from research_summary fields.
      If a paper lacks a research_summary (e.g. AI enrichment failed), that
      paper contributes nothing to the prompt and the LLM only sees N-1 papers.
      FIX: Fall back to abstract, then title, so every paper is always present.

    BUG 2 — "Cannot identify a theme" response
      compare_papers_narrative() has no cluster signal — it sends raw paper
      content and lets the LLM decide. When papers share only semantic embedding
      similarity (no entity overlap), the LLM correctly says it can't find a
      theme. But to the user this looks like a bug.
      FIX: Always inject cluster_label + signal_type into the prompt as a
      grounding anchor. Instruct the LLM it MUST synthesize a connection.

    BUG 3 — Wrong paper count mentioned in summary
      If paper 3 of 3 has no research_summary, the prompt silently contains
      only 2 paper blocks. The LLM then talks about "both papers" instead of
      "all 3 papers". FIX: covered by BUG 1 fix above.

    Parameters
    ──────────
    papers_in_cluster : list of paper dicts (title, abstract, entities, topics,
                        research_summary). Must be the actual DB paper records.
    cluster_label     : the auto-generated label from _label_cluster(), e.g.
                        "rag · fine-tuning · named entity recognition".
    signal_type       : "strong" | "weak" | "none" — controls how the LLM
                        frames the connection strength.
    """
    n = len(papers_in_cluster)
    if n == 0:
        return "No papers in this cluster."
    if n == 1:
        p = papers_in_cluster[0]
        return (
            f'This cluster contains a single paper: "{p.get("title", "Unknown")}". '
            f"It is grouped under the theme '{cluster_label}'."
        )

    from app.entity_normalizer import normalize_entity_list

    # ── Build paper blocks — BUG 1 FIX: always fall back to abstract/title ────
    paper_blocks: List[str] = []
    for i, p in enumerate(papers_in_cluster):
        rs = p.get("research_summary") or {}

        # Problem field: research_summary → abstract → title
        problem = (
            rs.get("problem_definition") or
            p.get("abstract") or
            f"Research paper: {p.get('title', 'Unknown')}"
        )

        # Methodology field: research_summary → entity methods → topics
        methodology = (
            rs.get("methodology") or
            ", ".join(normalize_entity_list(
                (p.get("entities") or {}).get("methods") or []
            )) or
            ", ".join(normalize_entity_list(p.get("topics") or [])) or
            "N/A"
        )

        topics_str = ", ".join(
            normalize_entity_list(p.get("topics") or [])[:5]
        ) or "N/A"

        paper_blocks.append(
            f'Paper {i + 1}: "{p.get("title", "Unknown")}"\n'
            f"  Topic/Problem: {problem[:400]}\n"
            f"  Methods: {methodology[:200]}\n"
            f"  Topics: {topics_str}"
        )

    papers_text = "\n\n".join(paper_blocks)

    # ── BUG 2 FIX: signal-aware prompt — inject cluster label as anchor ────────
    if signal_type == "strong":
        connection_context = (
            f'These {n} papers were clustered together because they share '
            f'research entities under the theme: "{cluster_label}". '
            f"They have overlapping methods, datasets, or topics."
        )
        synthesis_instruction = (
            f"Explain in 3-4 sentences what research theme unites all {n} papers. "
            f"Be specific about the shared methods or problems."
        )
    elif signal_type == "weak":
        connection_context = (
            f'These {n} papers were grouped under "{cluster_label}" based on '
            f"semantic similarity of their content embeddings. They may not share "
            f"exact methods or datasets, but occupy a related area of research."
        )
        synthesis_instruction = (
            f"Explain in 3-4 sentences what broader research area connects all {n} papers. "
            f"Note explicitly that their connection is thematic rather than methodological."
        )
    else:  # "none" — pure semantic clustering, no entity overlap
        connection_context = (
            f'These {n} papers were algorithmically grouped together under '
            f'"{cluster_label}" based on the semantic similarity of their '
            f"text embeddings. They may address related problems from different angles."
        )
        synthesis_instruction = (
            f"Identify and explain in 3-4 sentences the broader research area or "
            f"application domain that connects all {n} papers. "
            f"You MUST find a connection — even a broad one such as a shared application "
            f"domain, evaluation paradigm, or research question. "
            f"Do NOT say you cannot identify a theme."
        )

    # BUG 3 FIX: explicitly tell the LLM how many papers there are and require
    # all of them to be mentioned, so none get silently dropped.
    prompt = f"""You are a research analyst helping users understand their paper library.

{connection_context}

Here are all {n} papers in this cluster:

{papers_text}

Task: {synthesis_instruction}

STRICT RULES:
- Mention all {n} papers (you may refer to them by a short title fragment).
- Do NOT say "based only on the provided context I cannot identify..."
- Do NOT say "it is not possible to identify..."
- You MUST synthesize a connection. If the link is only broad, state it clearly.
- Write flowing prose (no bullet points). 3-4 sentences maximum.

Research theme summary:"""

    result = _call_gemini(prompt) or _call_ollama(prompt)

    if result and result.strip():
        return result.strip()

    # ── Deterministic fallback — never returns "cannot identify" ──────────────
    titles_fragment = ", ".join(
        f'"{p.get("title", "Unknown")[:50]}…"'
        for p in papers_in_cluster
    )
    all_topics = normalize_entity_list(
        [t for p in papers_in_cluster for t in (p.get("topics") or [])]
    )
    topic_str = ", ".join(all_topics[:5]) or cluster_label
    return (
        f"These {n} papers are grouped under the theme '{cluster_label}'. "
        f"The cluster includes {titles_fragment}. "
        f"They share research focus around: {topic_str}."
    )


# ─── Section summary ──────────────────────────────────────────────────────────

def generate_section_summary(
    section_title: str, section_content: str, paper_title: str = ""
) -> str:
    section_lower = section_title.lower()

    if any(w in section_lower for w in ["abstract"]):
        focus = "Write 2-3 sentences: core contribution, method, key result with numbers."
    elif any(w in section_lower for w in ["introduction", "intro"]):
        focus = "Write 3-4 sentences: problem, prior-work limits, proposed solution, contributions."
    elif any(w in section_lower for w in ["related", "background", "prior"]):
        focus = "Write 3-4 sentences: prior-work categories, key methods cited, positioning."
    elif any(w in section_lower for w in ["method", "model", "approach", "architecture", "framework"]):
        focus = "Write 4-5 sentences: step-by-step approach, architecture, key design choices."
    elif any(w in section_lower for w in ["experiment", "result", "evaluation", "benchmark"]):
        focus = "Write 3-4 sentences: exact performance numbers, baselines, ablation results."
    elif any(w in section_lower for w in ["conclusion", "discussion", "future"]):
        focus = "Write 2-3 sentences: main contributions, limitations, future directions."
    else:
        focus = "Write 3 technical sentences covering key points of this section."

    prompt = f"""Summarize the "{section_title}" section of "{paper_title}" for researchers.

SECTION CONTENT:
{section_content[:1500]}

{focus}

Rules: Be specific and technical. Include model names and numbers. Prose only. No verbatim copying.

Summary:"""

    response = _call_gemini(prompt)
    if response:
        return response.strip()
    return _simple_extractive_summary(section_content)


def answer_question(
    question: str,
    search_results: Optional[List[Dict[str, Any]]] = None,
    chat_history: Optional[List[Dict]] = None,
    context_papers: Optional[List[Dict]] = None,
    # Legacy positional args kept for any direct callers
    context_chunks: Optional[List[str]] = None,
    paper_titles: Optional[List[str]] = None,
) -> str:
    # Build context from search_results (new calling convention from main.py)
    if search_results is not None:
        chunks     = [r.get("text", "") for r in search_results[:5]]
        titles     = list({r.get("paper_id", "") for r in search_results})
        context    = "\n\n---\n\n".join(chunks)
        papers_str = ", ".join(titles[:3]) or "uploaded papers"
    else:
        # Legacy path
        context    = "\n\n---\n\n".join((context_chunks or [])[:5])
        papers_str = ", ".join(set((paper_titles or [])[:3])) or "uploaded papers"

    # Build optional graph context from related papers
    graph_note = ""
    if context_papers:
        titles_list = [p.get("title", "") for p in context_papers if p.get("title")]
        if titles_list:
            graph_note = f"\nRelated papers in library: {', '.join(titles_list[:3])}"

    # Build chat history string
    history_str = ""
    if chat_history:
        history_lines = []
        for msg in chat_history[-6:]:  # last 3 turns
            role    = msg.get("role", "user").capitalize()
            content = msg.get("content", "")[:300]
            history_lines.append(f"{role}: {content}")
        if history_lines:
            history_str = "\nConversation so far:\n" + "\n".join(history_lines) + "\n"

    prompt = f"""You are a research assistant. Answer based ONLY on the context below.

Context from: {papers_str}{graph_note}
{history_str}
---
{context[:3500]}
---

Question: {question}

Give a precise, technical answer with specific details. Include numbers and model names where relevant."""

    response = _call_gemini(prompt)
    if response:
        return response.strip()
    return "I couldn't generate an answer. Please review the source chunks below for relevant context."


def compare_papers(papers_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Thin wrapper called by main.py /api/compare endpoint.
    Delegates to compare_papers_deep() for full structured output.
    """
    return compare_papers_deep(papers_data, graph_context=None)


def generate_figure_summary(
    label: str, caption: str, item_type: str, paper_title: str = ""
) -> str:
    type_instructions = {
        "table":     "Explain what this table measures, rows/columns, key results, conclusions.",
        "figure":    "Describe what this figure visualizes, axes/components, key trends, insight.",
        "algorithm": "Explain what this algorithm does, its inputs/outputs, and role in the paper.",
    }
    instruction = type_instructions.get(item_type, "Explain what this element shows and its significance.")

    prompt = f"""Explain this {item_type} from "{paper_title}" to a researcher.

{label} Caption: {caption}

{instruction}

Write 3-5 sentences. Be specific. Reference numbers from the caption."""

    response = _call_gemini(prompt)
    if response:
        return response.strip()
    return f"{label}: {caption}"


# ─── Fallback extraction ──────────────────────────────────────────────────────

def _fallback_extraction(title: str, abstract: str) -> Dict[str, Any]:
    """Pure keyword/regex extraction — zero LLM calls. FIXED: normalizes results."""
    from app.entity_normalizer import normalize_entity_list

    method_keywords = [
        "transformer", "bert", "gpt", "lstm", "cnn", "attention", "gradient descent",
        "neural network", "random forest", "svm", "reinforcement learning", "fine-tuning",
        "contrastive learning", "diffusion", "gan", "retrieval", "rag", "embedding",
        "self-attention", "multi-head attention", "encoder", "decoder", "feedforward",
    ]
    dataset_keywords = [
        "imagenet", "coco", "squad", "glue", "mnist", "cifar", "wikipedia",
        "commonvoice", "librispeech", "ms marco", "natural questions", "wmt",
        "superglue", "winogrande", "hellaswag",
    ]

    text_lower = (title + " " + abstract).lower()

    methods  = normalize_entity_list([k for k in method_keywords  if k in text_lower][:6])
    datasets = normalize_entity_list([k.upper() for k in dataset_keywords if k in text_lower][:4])

    stop_words = {
        "a", "an", "the", "for", "on", "in", "of", "and", "with",
        "using", "via", "from", "this", "we", "our",
    }
    topics = normalize_entity_list([
        w for w in title.lower().split()
        if w not in stop_words and len(w) > 4
    ][:5])

    summary_text = abstract if abstract else f"Research paper: {title}"

    return {
        "research_summary": {
            "problem_definition":        summary_text,
            "background_and_prior_work": "",
            "methodology":               ", ".join(methods) if methods else "",
            "technical_novelty":         "",
            "experimental_setup":        ", ".join(datasets) if datasets else "",
            "results_and_analysis":      "",
            "limitations":               "",
            "conclusion":                "",
        },
        "research_gap": {
            "gap_identification": "", "remaining_limitations": "",
            "methodological_weaknesses": "", "future_directions": [],
        },
        "entities": {
            "methods":  methods or [],
            "models":   [],
            "datasets": datasets,
            "tasks":    [],
            "topics":   topics or ["machine learning"],
        },
        "topics":            topics or ["machine learning"],
        "section_summaries": {},
    }


def _simple_extractive_summary(text: str, num_sentences: int = 4) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    filtered  = [s.strip() for s in sentences if len(s.split()) > 8]
    return " ".join(filtered[:num_sentences]) if filtered else text[:300]