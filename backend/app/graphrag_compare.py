"""
graphrag_compare.py
────────────────────────────────────────────────────────────────────────────────
True GraphRAG comparison engine for Lens.

ARCHITECTURE:
  User selects papers
   → fetch from DB (PostgreSQL)
   → compress each paper to compact structured block
   → build_graph_context() → entity overlap as primary signal
   → pull graph similarity edges from Neo4j
   → pull 2 precise chunks per paper from Qdrant (grounding evidence)
   → build compact combined input (< 1 500 tokens total)
   → call LLM once with GraphRAG-first prompt
   → cache result keyed by frozenset(paper_ids)
   → return structured comparison

Key principle: graph_context is the PRIMARY input. LLM explains it — it does
NOT discover relationships from scratch. If no graph signal exists, the LLM
explicitly states "weakly related" rather than hallucinating.

Public API:
    graphrag_compare(paper_ids, papers, db, user_id) -> dict
"""

from __future__ import annotations

import hashlib
import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple

# ── Comparison cache ───────────────────────────────────────────────────────────
_comparison_cache: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL_SECONDS = 3600


def _cache_key(paper_ids: List[str]) -> str:
    canonical = ",".join(sorted(paper_ids))
    return hashlib.md5(canonical.encode()).hexdigest()


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    entry = _comparison_cache.get(key)
    if entry and (time.time() - entry["_ts"]) < _CACHE_TTL_SECONDS:
        return entry["data"]
    return None


def _cache_set(key: str, data: Dict[str, Any]) -> None:
    _comparison_cache[key] = {"data": data, "_ts": time.time()}


# ── Step 1: compress each paper to compact block ───────────────────────────────

def _compress_paper(p: Any, idx: int) -> Dict[str, Any]:
    rs  = p.research_summary or {}
    rg  = p.research_gap     or {}
    ent = p.entities         or {}

    def _trim(text: str, chars: int = 220) -> str:
        if not text:
            return ""
        text = text.strip()
        return text[:chars] + "…" if len(text) > chars else text

    problem     = _trim(rs.get("problem_definition")    or p.abstract or "")
    methodology = _trim(rs.get("methodology")           or "")
    novelty     = _trim(rs.get("technical_novelty")     or "")
    results     = _trim(rs.get("results_and_analysis")  or "")
    limitations = _trim(rs.get("limitations")           or rg.get("remaining_limitations") or "")
    gap         = _trim(rg.get("gap_identification")    or "")

    methods  = (ent.get("methods",  []) or [])[:4]
    datasets = (ent.get("datasets", []) or [])[:3]
    models   = (ent.get("models",   []) or [])[:3]
    tasks    = (ent.get("tasks",    []) or [])[:3]

    return {
        "idx":        idx,
        "paper_id":   p.paper_id,
        "title":      p.title,
        "authors":    (p.authors or [])[:3],
        "topics":     (p.topics  or [])[:4],
        "problem":    problem,
        "methodology": methodology,
        "novelty":    novelty,
        "results":    results,
        "limitations": limitations,
        "gap":        gap,
        "methods":    methods,
        "datasets":   datasets,
        "models":     models,
        "tasks":      tasks,
    }


# ── Step 2: build graph context (entity overlap) ──────────────────────────────

def _build_paper_dicts_for_graph_context(papers: List[Any]) -> List[dict]:
    """Convert SQLAlchemy Paper ORM objects to plain dicts for graph_context."""
    result = []
    for p in papers:
        result.append({
            "paper_id": p.paper_id,
            "title":    p.title,
            "entities": p.entities or {},
            "topics":   p.topics   or [],
        })
    return result


# ── Step 3: pull Neo4j similarity edges ───────────────────────────────────────

def _get_neo4j_similarity_edges(
    paper_ids: List[str],
    user_id: int,
) -> List[dict]:
    """
    Pull or compute SIMILAR_TO edge scores from Neo4j for all pairs.
    Writes missing edges on-the-fly using the hybrid similarity model.
    """
    edges: List[dict] = []

    try:
        from app.graph_db import get_driver
        driver = get_driver()

        with driver.session() as session:
            for i, pid_a in enumerate(paper_ids):
                for pid_b in paper_ids[i + 1:]:
                    rec = session.run(
                        """
                        MATCH (a:Paper {paper_id: $pid_a})-[r:SIMILAR_TO]->(b:Paper {paper_id: $pid_b})
                        RETURN r.score AS score, r.shared_topics AS st,
                               r.shared_methods AS sm, r.explanation AS exp
                        """,
                        pid_a=pid_a, pid_b=pid_b,
                    ).single()

                    if rec:
                        edges.append({
                            "paper_a":        pid_a,
                            "paper_b":        pid_b,
                            "score":          round(float(rec["score"] or 0), 3),
                            "shared_topics":  list(rec["st"]  or [])[:5],
                            "shared_methods": list(rec["sm"]  or [])[:5],
                            "explanation":    rec["exp"] or "",
                        })
                    else:
                        # Compute on-the-fly
                        emb_rec = session.run(
                            """
                            MATCH (a:Paper {paper_id: $pid_a}), (b:Paper {paper_id: $pid_b})
                            WHERE a.embedding IS NOT NULL AND b.embedding IS NOT NULL
                            RETURN a.embedding AS emb_a, b.embedding AS emb_b
                            """,
                            pid_a=pid_a, pid_b=pid_b,
                        ).single()

                        if emb_rec:
                            import numpy as np
                            ea, eb = emb_rec["emb_a"], emb_rec["emb_b"]
                            score = 0.0
                            explanation = ""
                            try:
                                from app.similarity_model import score_pair
                                sh_t = session.run(
                                    """MATCH (a:Paper {paper_id:$a})-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(b:Paper {paper_id:$b})
                                       RETURN collect(DISTINCT t.name) AS v""",
                                    a=pid_a, b=pid_b,
                                ).single()
                                sh_m = session.run(
                                    """MATCH (a:Paper {paper_id:$a})-[:USES_METHOD]->(m:Method)<-[:USES_METHOD]-(b:Paper {paper_id:$b})
                                       RETURN collect(DISTINCT m.name) AS v""",
                                    a=pid_a, b=pid_b,
                                ).single()
                                sh_d = session.run(
                                    """MATCH (a:Paper {paper_id:$a})-[:USES_DATASET]->(d:Dataset)<-[:USES_DATASET]-(b:Paper {paper_id:$b})
                                       RETURN collect(DISTINCT d.name) AS v""",
                                    a=pid_a, b=pid_b,
                                ).single()
                                shared_topics   = list(sh_t["v"] or [])[:5] if sh_t else []
                                shared_methods  = list(sh_m["v"] or [])[:5] if sh_m else []
                                shared_datasets = list(sh_d["v"] or [])[:3] if sh_d else []
                                pa_d = {"paper_id": pid_a, "entities": {"methods": shared_methods, "datasets": shared_datasets}, "topics": shared_topics}
                                pb_d = {"paper_id": pid_b, "entities": {"methods": shared_methods, "datasets": shared_datasets}, "topics": shared_topics}
                                hybrid = score_pair(pa_d, pb_d, ea, eb)
                                score = hybrid.similarity_score
                                explanation = hybrid.explanation
                            except Exception:
                                va = np.array(ea, dtype=np.float32)
                                vb = np.array(eb, dtype=np.float32)
                                na, nb = np.linalg.norm(va), np.linalg.norm(vb)
                                score = float(np.dot(va, vb) / (na * nb + 1e-9)) if na > 0 and nb > 0 else 0.0

                            edges.append({
                                "paper_a": pid_a, "paper_b": pid_b,
                                "score": round(score, 3),
                                "shared_topics": [], "shared_methods": [],
                                "explanation": explanation,
                            })

                            if score > 0:
                                session.run(
                                    """
                                    MATCH (p1:Paper {paper_id:$id1}),(p2:Paper {paper_id:$id2})
                                    MERGE (p1)-[r:SIMILAR_TO]->(p2) SET r.score=$sc,r.explanation=$exp
                                    MERGE (p2)-[r2:SIMILAR_TO]->(p1) SET r2.score=$sc,r2.explanation=$exp
                                    """,
                                    id1=pid_a, id2=pid_b, sc=round(score, 4), exp=explanation,
                                )

    except Exception as e:
        print(f"⚠️ GraphRAG: Neo4j edge retrieval failed (non-critical): {e}")

    return edges


# ── Step 4: pull Qdrant support chunks ────────────────────────────────────────

def _get_qdrant_support_chunks(papers: List[Any], user_id: int) -> Dict[str, List[str]]:
    support: Dict[str, List[str]] = {}
    try:
        from app.vector_db import search_similar
        from app.embeddings import generate_single_embedding

        for p in papers:
            query = f"methodology approach results findings {p.title}"
            emb   = generate_single_embedding(query)
            chunks = search_similar(query_embedding=emb, user_id=user_id, paper_id=p.paper_id, top_k=4)

            priority_kw = {"methodology", "methods", "results", "experiments",
                           "evaluation", "approach", "model", "framework"}
            priority = [c for c in chunks if any(kw in c["section"].lower() for kw in priority_kw)]
            fallback = [c for c in chunks if c not in priority]
            selected = (priority + fallback)[:2]
            support[p.paper_id] = [c["text"][:300].strip() for c in selected]

    except Exception as e:
        print(f"⚠️ GraphRAG: Qdrant chunk retrieval failed (non-critical): {e}")

    return support


# ── Step 5: build compact LLM input ──────────────────────────────────────────

def _build_llm_input(
    compressed:    List[Dict[str, Any]],
    graph_ctx_str: str,
    neo4j_edges:   List[dict],
    support:       Dict[str, List[str]],
) -> str:
    """
    Build the compact LLM input.
    Order: graph_context (primary) → paper summaries → Qdrant evidence → Neo4j scores.
    Total target: < 1 500 tokens.
    """
    lines: List[str] = []

    # ── Graph context FIRST (primary signal) ─────────────────────────────────
    lines.append(graph_ctx_str)
    lines.append("")

    # ── Paper blocks ─────────────────────────────────────────────────────────
    for c in compressed:
        n = c["idx"]
        lines.append(f"=== P{n}: {c['title']} ===")
        if c["problem"]:    lines.append(f"Problem: {c['problem']}")
        if c["methodology"]:lines.append(f"Method: {c['methodology']}")
        if c["novelty"]:    lines.append(f"Novelty: {c['novelty']}")
        if c["results"]:    lines.append(f"Results: {c['results']}")
        if c["limitations"]:lines.append(f"Limits: {c['limitations']}")
        if c["methods"]:    lines.append(f"Techniques: {', '.join(c['methods'])}")
        if c["datasets"]:   lines.append(f"Datasets: {', '.join(c['datasets'])}")
        if c["models"]:     lines.append(f"Models: {', '.join(c['models'])}")
        if c["tasks"]:      lines.append(f"Tasks: {', '.join(c['tasks'])}")

        for j, chunk in enumerate(support.get(c["paper_id"], [])):
            lines.append(f"Evidence{j+1}: {chunk}")
        lines.append("")

    # ── Neo4j similarity scores ───────────────────────────────────────────────
    if neo4j_edges:
        lines.append("=== GRAPH SIMILARITY SCORES (hybrid model) ===")
        for edge in neo4j_edges:
            pa_idx = next((c["idx"] for c in compressed if c["paper_id"] == edge["paper_a"]), "?")
            pb_idx = next((c["idx"] for c in compressed if c["paper_id"] == edge["paper_b"]), "?")
            lines.append(f"P{pa_idx}↔P{pb_idx}: {edge['score']:.0%}")

    return "\n".join(lines)


# ── Step 6: call LLM with true GraphRAG prompt ────────────────────────────────

def _llm_compare(llm_input: str, n_papers: int, has_graph_signal: bool) -> Dict[str, Any]:
    """
    True GraphRAG LLM call.

    The prompt explicitly instructs the model to:
    - Use graph_context as PRIMARY source (entity overlap = ground truth)
    - Use paper summaries as SECONDARY support
    - NOT hallucinate relationships not present in graph_context
    - State "weakly related" if no shared entities exist
    - Return structured JSON with graph_based_similarity and confidence fields
    """
    from app import llm_service

    sw_template = "\n".join([
        f'"paper{i}": {{"strengths": ["s1","s2"], "weaknesses": ["w1","w2"]}}'
        for i in range(1, n_papers + 1)
    ])
    uc_template = "\n".join([
        f'"paper{i}": "best use case"' for i in range(1, n_papers + 1)
    ])

    no_signal_instruction = (
        "The graph context reports NO shared entities. "
        "You MUST state these papers are weakly related based on semantic similarity only. "
        "Do NOT invent shared methods or topics."
        if not has_graph_signal
        else
        "The graph context shows entity overlap. Base your comparison primarily on these signals."
    )

    prompt = f"""You are a research assistant performing graph-based paper comparison.

INSTRUCTIONS:
- Use the GRAPH CONTEXT section as PRIMARY evidence (entity overlap = ground truth).
- Use paper summaries as SECONDARY support to add detail.
- Use Qdrant evidence blocks as grounding — cite numbers/results from them.
- Do NOT hallucinate relationships not present in the graph context.
- If graph_based_similarity is low or no shared entities exist, say so explicitly.
- {no_signal_instruction}

{llm_input}

Return ONLY valid JSON, no markdown:
{{
  "comparison_table": {{
    "problem":     [{", ".join([f'"P{i} problem in 1 sentence"' for i in range(1, n_papers+1)])}],
    "methodology": [{", ".join([f'"P{i} core method in 1 sentence"' for i in range(1, n_papers+1)])}],
    "datasets":    [{", ".join([f'"P{i} datasets or N/A"' for i in range(1, n_papers+1)])}],
    "results":     [{", ".join([f'"P{i} key result with numbers if available"' for i in range(1, n_papers+1)])}]
  }},
  "key_differences": ["graph-grounded difference 1", "difference 2", "difference 3"],
  "problem_solution_relationship": "3 sentences grounded in graph context: how papers relate, share methods, or diverge",
  "graph_based_similarity": "1-2 sentences: what the entity overlap tells us about similarity. If none: state weakly related.",
  "confidence": "high | medium | low — based on how much graph signal is available",
  "strengths_vs_weaknesses": {{
    {sw_template}
  }},
  "best_use_cases": {{
    {uc_template}
  }},
  "research_evolution": "2-3 sentences: how these papers advance the field, grounded in shared/unique methods"
}}"""

    response = llm_service._call_gemini(prompt)

    if response:
        try:
            clean = re.sub(r"```(?:json)?\s*|\s*```", "", response).strip()
            result = json.loads(clean)
            return _validate_llm_result(result, n_papers)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", response, re.DOTALL)
            if m:
                try:
                    result = json.loads(m.group())
                    return _validate_llm_result(result, n_papers)
                except Exception:
                    pass

    return {}


def _validate_llm_result(result: dict, n: int) -> dict:
    defaults = {
        "comparison_table":              {"problem": [], "methodology": [], "datasets": [], "results": []},
        "key_differences":               [],
        "problem_solution_relationship": "",
        "graph_based_similarity":        "",
        "confidence":                    "low",
        "strengths_vs_weaknesses":       {},
        "best_use_cases":                {},
        "research_evolution":            "",
    }
    for k, v in defaults.items():
        if k not in result:
            result[k] = v
    return result


# ── Step 7: structural fallback (zero LLM) ────────────────────────────────────

def _structural_fallback(
    compressed:  List[Dict[str, Any]],
    graph_ctx:   Any,       # GraphContext
    neo4j_edges: List[dict],
) -> Dict[str, Any]:
    """Pure structural comparison from graph context + compressed fields. Zero LLM calls."""
    from app.graph_context import GraphContext

    table = {
        "problem":     [c["problem"]     or "N/A" for c in compressed],
        "methodology": [c["methodology"] or "N/A" for c in compressed],
        "datasets":    [", ".join(c["datasets"]) if c["datasets"] else "N/A" for c in compressed],
        "results":     [c["results"]     or "N/A" for c in compressed],
    }

    diffs: List[str] = []
    for pid, methods in graph_ctx.unique_methods.items():
        idx = next((c["idx"] for c in compressed if c["paper_id"] == pid), "?")
        if methods:
            diffs.append(f"P{idx} uniquely uses: {', '.join(methods[:3])}")
    if graph_ctx.shared_methods:
        diffs.append(f"All papers share method(s): {', '.join(graph_ctx.shared_methods[:3])}")
    else:
        diffs.append("No shared techniques found in the knowledge graph")

    # Build relationship summary from graph
    if graph_ctx.connections:
        psr = ". ".join(graph_ctx.connections[:3]) + "."
    elif neo4j_edges:
        parts = []
        for edge in neo4j_edges:
            pa_idx = next((c["idx"] for c in compressed if c["paper_id"] == edge["paper_a"]), "?")
            pb_idx = next((c["idx"] for c in compressed if c["paper_id"] == edge["paper_b"]), "?")
            parts.append(f"P{pa_idx} and P{pb_idx} have hybrid similarity {edge['score']:.0%}")
        psr = ". ".join(parts) + "."
    else:
        psr = "Papers address related research areas with distinct methodological approaches."

    graph_sim = (
        f"Shared methods: {', '.join(graph_ctx.shared_methods[:3])}. " if graph_ctx.shared_methods else ""
    ) + (
        f"Shared topics: {', '.join(graph_ctx.shared_topics[:3])}. " if graph_ctx.shared_topics else ""
    ) + (
        "No shared entities detected — similarity is semantic only." if not graph_ctx.has_signal else ""
    )

    sw: Dict[str, Any] = {}
    for c in compressed:
        key = f"paper{c['idx']}"
        sw[key] = {
            "strengths":  [c["novelty"]] if c["novelty"] else ["Novel research contribution"],
            "weaknesses": [c["limitations"]] if c["limitations"] else ["Limitations not fully documented"],
        }

    uc: Dict[str, str] = {
        f"paper{c['idx']}": f"Researchers working on {', '.join(c['topics'][:2]) or c['title'][:40]}"
        for c in compressed
    }

    return {
        "comparison_table":              table,
        "key_differences":               diffs[:4],
        "problem_solution_relationship": psr,
        "graph_based_similarity":        graph_sim.strip(),
        "confidence":                    "high" if graph_ctx.has_signal else "low",
        "strengths_vs_weaknesses":       sw,
        "best_use_cases":                uc,
        "research_evolution":            (
            "These papers represent complementary advances: "
            + " and ".join(f"P{c['idx']} ({c['title'][:30]}…)" for c in compressed) + "."
        ),
        "_source": "structural_fallback",
    }


# ── Step 8: hybrid similarity scores ─────────────────────────────────────────

def _hybrid_similarity_scores(
    papers:      List[Any],
    neo4j_edges: List[dict],
) -> Dict[str, float]:
    """Prefer Neo4j edge scores; compute on-the-fly for missing pairs."""
    scores: Dict[str, float] = {}
    paper_ids = [p.paper_id for p in papers]

    for edge in neo4j_edges:
        ia = paper_ids.index(edge["paper_a"]) + 1
        ib = paper_ids.index(edge["paper_b"]) + 1
        scores[f"paper{ia}_vs_paper{ib}"] = edge["score"]

    for i, pa in enumerate(papers):
        for j, pb in enumerate(papers):
            if j <= i:
                continue
            key = f"paper{i+1}_vs_paper{j+1}"
            if key in scores:
                continue
            try:
                from app.embeddings import generate_single_embedding
                from app.similarity_model import score_pair
                ea = generate_single_embedding(f"{pa.title}. {pa.abstract or ''}")
                eb = generate_single_embedding(f"{pb.title}. {pb.abstract or ''}")
                pa_d = {"paper_id": pa.paper_id, "entities": pa.entities or {}, "topics": pa.topics or []}
                pb_d = {"paper_id": pb.paper_id, "entities": pb.entities or {}, "topics": pb.topics or []}
                result = score_pair(pa_d, pb_d, ea, eb)
                scores[key] = round(result.similarity_score, 4)
            except Exception:
                pass

    return scores


# ── Public entry point ────────────────────────────────────────────────────────

def graphrag_compare(
    paper_ids: List[str],
    papers:    List[Any],
    user_id:   int,
    db:        Any,
) -> Dict[str, Any]:
    """
    Full GraphRAG comparison pipeline.

    Pipeline:
      1. Compress papers to structured dicts
      2. build_graph_context() → entity overlap (PRIMARY signal)
      3. Pull Neo4j SIMILAR_TO edges (similarity scores)
      4. Pull Qdrant support chunks (grounding evidence)
      5. Build compact LLM input (graph_context first, then summaries)
      6. Call LLM with GraphRAG-first prompt
      7. Structural fallback if LLM fails / quota exceeded
      8. Compute hybrid similarity scores

    Returns dict with:
      papers_data, comparison_table, key_differences,
      problem_solution_relationship, graph_based_similarity, confidence,
      strengths_vs_weaknesses, best_use_cases, research_evolution,
      graph_context, graph_signals, similarity_scores, source
    """
    from app import llm_service
    from app.graph_context import build_graph_context, graph_context_to_str

    if len(papers) < 2:
        raise ValueError("Need at least 2 papers")

    ck = _cache_key(paper_ids)
    cached = _cache_get(ck)
    if cached:
        print(f"✅ GraphRAG compare: cache hit for {ck[:8]}")
        result = dict(cached)
        result["source"] = "graphrag_cached"
        return result

    print(f"🔬 GraphRAG compare: {len(papers)} papers — starting pipeline")

    # ── Step 1: compress ───────────────────────────────────────────────────────
    compressed = [_compress_paper(p, i + 1) for i, p in enumerate(papers)]

    # ── Step 2: build graph context (entity overlap — PRIMARY signal) ──────────
    paper_dicts = _build_paper_dicts_for_graph_context(papers)
    graph_ctx   = build_graph_context(paper_dicts, top_k=3)
    graph_ctx_str = graph_context_to_str(graph_ctx)
    print(f"  📊 Graph context: shared_methods={graph_ctx.shared_methods}, "
          f"shared_topics={graph_ctx.shared_topics}, has_signal={graph_ctx.has_signal}")

    # ── Step 3: Neo4j similarity edges ────────────────────────────────────────
    neo4j_edges = _get_neo4j_similarity_edges(paper_ids, user_id)
    print(f"  🔗 Neo4j edges: {len(neo4j_edges)}")

    # ── Step 4: Qdrant support chunks ─────────────────────────────────────────
    support = _get_qdrant_support_chunks(papers, user_id)
    print(f"  🔍 Qdrant chunks: {len(support)} papers")

    # ── Step 5 + 6: build input and call LLM ──────────────────────────────────
    llm_input  = _build_llm_input(compressed, graph_ctx_str, neo4j_edges, support)
    print(f"  📝 LLM input: {len(llm_input)} chars")

    llm_result: Dict[str, Any] = {}
    source = "graphrag_structural"

    llm_result = _llm_compare(llm_input, len(papers), graph_ctx.has_signal)
    if llm_result:
        source = "graphrag_llm"
        print("  🤖 LLM comparison: success")
    else:
        print("  ⚠️  LLM failed — using structural fallback")

    # ── Step 7: structural fallback ───────────────────────────────────────────
    if not llm_result:
        llm_result = _structural_fallback(compressed, graph_ctx, neo4j_edges)

    # ── Step 8: hybrid similarity scores ──────────────────────────────────────
    sim_scores = _hybrid_similarity_scores(papers, neo4j_edges)

    # ── Assemble ──────────────────────────────────────────────────────────────
    papers_data = [
        {
            "paper_id":    c["paper_id"],
            "title":       c["title"],
            "authors":     c["authors"],
            "topics":      c["topics"],
            "problem":     c["problem"],
            "methodology": c["methodology"],
            "results":     c["results"],
            "methods":     c["methods"],
            "datasets":    c["datasets"],
            "models":      c["models"],
            "tasks":       c["tasks"],
        }
        for c in compressed
    ]

    result = {
        "papers":                        papers_data,
        "comparison_table":              llm_result.get("comparison_table", {}),
        "key_differences":               llm_result.get("key_differences", []),
        "problem_solution_relationship": llm_result.get("problem_solution_relationship", ""),
        "graph_based_similarity":        llm_result.get("graph_based_similarity", ""),
        "confidence":                    llm_result.get("confidence", "low"),
        "strengths_vs_weaknesses":       llm_result.get("strengths_vs_weaknesses", {}),
        "best_use_cases":                llm_result.get("best_use_cases", {}),
        "research_evolution":            llm_result.get("research_evolution", ""),
        # Graph data for frontend display
        "graph_context":                 graph_ctx.to_dict(),
        "graph_signals": {
            "similarity_edges": neo4j_edges,
            "shared_methods":   graph_ctx.shared_methods,
            "shared_datasets":  graph_ctx.shared_datasets,
            "shared_topics":    graph_ctx.shared_topics,
            "shared_models":    graph_ctx.shared_models,
            "unique_methods":   graph_ctx.unique_methods,
            "connections":      graph_ctx.connections,
            "has_signal":       graph_ctx.has_signal,
        },
        "similarity_scores": sim_scores or None,
        "source":            source,
    }

    _cache_set(ck, result)
    print(f"✅ GraphRAG compare done — source={source}, has_signal={graph_ctx.has_signal}")
    return result