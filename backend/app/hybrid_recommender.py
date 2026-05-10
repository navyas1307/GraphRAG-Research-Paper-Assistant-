"""
hybrid_recommender.py
─────────────────────
True hybrid recommendation engine that merges:
  • Qdrant  → chunk-level vector similarity  (alpha = 0.6)
  • Neo4j   → paper-level graph similarity   (beta  = 0.4)

Public API
──────────
  get_hybrid_recommendations(paper_id, user_id, db, top_k=10)
    → List[HybridRecommendation]

  Each result carries:
    paper_id, title, authors, abstract, topics,
    vector_score, graph_score, final_score,
    shared_topics, shared_methods, relevance_explanation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

# ── tuneable constants ─────────────────────────────────────────────────────────
ALPHA = 0.6          # weight for vector score
BETA  = 0.4          # weight for graph score
assert math.isclose(ALPHA + BETA, 1.0), "ALPHA + BETA must equal 1.0"

VECTOR_TOP_CHUNKS    = 50   # how many Qdrant chunks to pull per query
CANDIDATE_PAPERS     = 10   # max candidate papers from vector stage
CHUNK_AGGREGATE      = "max_avg3"   # "max" | "avg" | "max_avg3"
NEAR_DUPLICATE_THRESH = 0.9  # cosine sim above which we deduplicate
TOPIC_BOOST          = 0.05  # bonus per shared topic (capped at 0.10)
METHOD_BOOST         = 0.03  # bonus per shared method (capped at 0.06)
MAX_TOPIC_BOOST      = 0.10
MAX_METHOD_BOOST     = 0.06


# ── result dataclass ───────────────────────────────────────────────────────────

@dataclass
class HybridRecommendation:
    paper_id:   str
    title:      str
    authors:    List[str]
    abstract:   str
    topics:     List[str]

    vector_score: float          # normalised 0-1
    graph_score:  float          # normalised 0-1
    final_score:  float          # ALPHA*vector + BETA*graph + boosts

    shared_topics:  List[str] = field(default_factory=list)
    shared_methods: List[str] = field(default_factory=list)
    relevance_explanation: str = ""

    # raw internals (useful for debugging / logging)
    raw_chunk_scores: List[float] = field(default_factory=list)
    raw_graph_score:  float = 0.0


# ── helpers ────────────────────────────────────────────────────────────────────

def _aggregate_chunk_scores(scores: List[float], method: str) -> float:
    """
    Collapse per-chunk scores into a single paper-level vector score.

    Strategies
    ----------
    max       – single highest chunk (favours papers with one very relevant chunk)
    avg       – average of all chunks (can be dragged down by noise)
    max_avg3  – average of the top-3 chunks (balances both concerns, default)
    """
    if not scores:
        return 0.0
    scores_sorted = sorted(scores, reverse=True)
    if method == "max":
        return scores_sorted[0]
    if method == "avg":
        return sum(scores) / len(scores)
    # max_avg3 (default)
    top3 = scores_sorted[:3]
    return sum(top3) / len(top3)


def _minmax_normalise(values: Dict[str, float]) -> Dict[str, float]:
    """
    Min-max normalise a dict of {paper_id: raw_score} to [0, 1].
    If all values are equal, returns 1.0 for all (avoid division by zero).
    """
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if math.isclose(lo, hi):
        return {k: 1.0 for k in values}
    span = hi - lo
    return {k: (v - lo) / span for k, v in values.items()}


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


# ── stage 1 : vector-based paper retrieval ────────────────────────────────────

def _vector_paper_scores(
    query_embedding: List[float],
    user_id: int,
    paper_id: str,          # exclude the source paper
    top_chunks: int = VECTOR_TOP_CHUNKS,
    candidate_papers: int = CANDIDATE_PAPERS,
) -> Dict[str, float]:
    """
    Pull top-k chunks from Qdrant, group by paper_id, aggregate.
    Returns {candidate_paper_id: raw_aggregated_score}.
    Excludes the source paper itself.
    """
    from app.vector_db import search_similar  # lazy import to avoid circular deps

    chunks = search_similar(
        query_embedding=query_embedding,
        user_id=user_id,
        paper_id=None,          # search across ALL papers for this user
        top_k=top_chunks,
    )

    # Group by paper
    paper_chunks: Dict[str, List[float]] = {}
    for chunk in chunks:
        pid = chunk["paper_id"]
        if pid == paper_id:     # skip source paper
            continue
        paper_chunks.setdefault(pid, []).append(chunk["score"])

    # Aggregate
    paper_scores = {
        pid: _aggregate_chunk_scores(scores, CHUNK_AGGREGATE)
        for pid, scores in paper_chunks.items()
    }

    # Return top N candidates
    top_candidates = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)
    return dict(top_candidates[:candidate_papers])


# ── stage 2 : graph-based retrieval ──────────────────────────────────────────

def _graph_paper_scores(
    paper_id: str,
    user_id: int,
) -> Tuple[Dict[str, float], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Fetch all SIMILAR_TO edges from Neo4j for the source paper.
    Returns:
      raw_scores      {candidate_paper_id: similarity_score}
      shared_topics   {candidate_paper_id: [topic, ...]}
      shared_methods  {candidate_paper_id: [method, ...]}
    """
    from app.graph_db import get_driver  # lazy import

    driver = get_driver()
    raw_scores:     Dict[str, float]       = {}
    shared_topics:  Dict[str, List[str]]   = {}
    shared_methods: Dict[str, List[str]]   = {}

    try:
        with driver.session() as session:
            records = session.run(
                """
                MATCH (src:Paper {paper_id: $paper_id})-[r:SIMILAR_TO]->(tgt:Paper {user_id: $user_id})
                WHERE tgt.paper_id <> $paper_id
                RETURN tgt.paper_id   AS pid,
                       r.score        AS score,
                       r.shared_topics  AS shared_topics,
                       r.shared_methods AS shared_methods
                """,
                paper_id=paper_id,
                user_id=user_id,
            ).data()

        for rec in records:
            pid = rec["pid"]
            if pid is None:
                continue
            raw_scores[pid]     = float(rec["score"] or 0.0)
            shared_topics[pid]  = rec["shared_topics"]  or []
            shared_methods[pid] = rec["shared_methods"] or []

    except Exception as e:
        print(f"⚠️  Neo4j graph retrieval failed (non-critical): {e}")

    return raw_scores, shared_topics, shared_methods


# ── stage 3 : merge + rank + diversity ────────────────────────────────────────

def _apply_diversity_filter(
    candidates: List[HybridRecommendation],
    near_dup_threshold: float = NEAR_DUPLICATE_THRESH,
) -> List[HybridRecommendation]:
    """
    Remove near-duplicate papers (cosine similarity > threshold on their
    embeddings).  Also enforces basic topic diversity: if N > 3 papers share
    the same top topic, keep only the top 3 of them by final_score.

    NOTE: Embedding comparison is done via Neo4j stored embeddings.
    Falls back gracefully if embeddings are unavailable.
    """
    from app.graph_db import get_driver  # lazy import

    # ── near-duplicate removal via stored embeddings ──
    try:
        driver = get_driver()
        paper_ids = [c.paper_id for c in candidates]
        with driver.session() as session:
            records = session.run(
                """
                MATCH (p:Paper)
                WHERE p.paper_id IN $pids AND p.embedding IS NOT NULL
                RETURN p.paper_id AS pid, p.embedding AS emb
                """,
                pids=paper_ids,
            ).data()

        emb_map: Dict[str, List[float]] = {r["pid"]: r["emb"] for r in records}

        kept: List[HybridRecommendation] = []
        for cand in candidates:
            emb_c = emb_map.get(cand.paper_id)
            if emb_c is None:
                kept.append(cand)
                continue
            is_dup = False
            for existing in kept:
                emb_e = emb_map.get(existing.paper_id)
                if emb_e is not None and _cosine_similarity(emb_c, emb_e) > near_dup_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(cand)
        candidates = kept

    except Exception as e:
        print(f"⚠️  Diversity filter (embedding stage) failed: {e}")

    # ── topic diversity cap: max 3 papers per dominant topic ──
    topic_counts: Dict[str, int] = {}
    diverse: List[HybridRecommendation] = []
    for cand in candidates:
        top_topic = cand.topics[0] if cand.topics else "__none__"
        if topic_counts.get(top_topic, 0) < 3:
            diverse.append(cand)
            topic_counts[top_topic] = topic_counts.get(top_topic, 0) + 1

    return diverse


def _build_explanation(
    vector_score: float,
    graph_score:  float,
    shared_topics: List[str],
    shared_methods: List[str],
) -> str:
    parts = []
    if vector_score > 0:
        parts.append(f"semantic match {vector_score:.0%}")
    if graph_score > 0:
        parts.append(f"graph similarity {graph_score:.0%}")
    if shared_topics:
        parts.append(f"shared topics: {', '.join(shared_topics[:3])}")
    if shared_methods:
        parts.append(f"shared methods: {', '.join(shared_methods[:2])}")
    return " · ".join(parts) if parts else "structural similarity"


# ── public entry point ─────────────────────────────────────────────────────────

def get_hybrid_recommendations(
    paper_id: str,
    user_id: int,
    db: Session,
    top_k: int = 10,
) -> List[HybridRecommendation]:
    """
    Main hybrid recommender.

    Steps
    ─────
    1. Build query embedding from source paper (title + abstract).
    2. Retrieve candidate papers from Qdrant (vector signal).
    3. Retrieve SIMILAR_TO edges from Neo4j (graph signal).
    4. Union both candidate sets.
    5. Normalise each score domain to [0, 1] independently.
    6. Compute final_score = ALPHA * vector + BETA * graph + boosts.
    7. Deduplicate near-identical papers and enforce topic diversity.
    8. Return top_k sorted descending.
    """
    from app.models import Paper
    from app.embeddings import generate_single_embedding

    # ── load source paper ──────────────────────────────────────────────────────
    source: Optional[Paper] = db.query(Paper).filter(
        Paper.paper_id == paper_id,
        Paper.user_id  == user_id,
    ).first()
    if not source:
        return []

    source_topics  = set(source.topics or [])
    source_methods = set((source.entities or {}).get("methods", []))

    # ── build query embedding ──────────────────────────────────────────────────
    abstract_text = (source.abstract or "")[:500]
    query_text    = f"{source.title}. {abstract_text}".strip()
    query_emb     = generate_single_embedding(query_text)

    # ── stage 1 : vector retrieval ─────────────────────────────────────────────
    raw_vector: Dict[str, float] = _vector_paper_scores(
        query_embedding=query_emb,
        user_id=user_id,
        paper_id=paper_id,
    )

    # ── stage 2 : graph retrieval ──────────────────────────────────────────────
    raw_graph, shared_topics_map, shared_methods_map = _graph_paper_scores(
        paper_id=paper_id,
        user_id=user_id,
    )

    # ── stage 3 : union candidate set ─────────────────────────────────────────
    all_candidate_ids = set(raw_vector) | set(raw_graph)
    if not all_candidate_ids:
        return []

    # ── normalise scores independently ────────────────────────────────────────
    norm_vector = _minmax_normalise(raw_vector)   # {pid: 0-1}
    norm_graph  = _minmax_normalise(raw_graph)    # {pid: 0-1}

    # ── compute final scores + boosts ─────────────────────────────────────────
    scored: List[Tuple[str, float, float, float]] = []  # (pid, vs, gs, final)
    for pid in all_candidate_ids:
        vs = norm_vector.get(pid, 0.0)
        gs = norm_graph.get(pid,  0.0)

        base_score = ALPHA * vs + BETA * gs

        # Bonus: shared topics
        s_topics  = shared_topics_map.get(pid,  [])
        s_methods = shared_methods_map.get(pid, [])
        topic_bonus  = min(len(s_topics)  * TOPIC_BOOST,  MAX_TOPIC_BOOST)
        method_bonus = min(len(s_methods) * METHOD_BOOST, MAX_METHOD_BOOST)

        final = base_score + topic_bonus + method_bonus
        scored.append((pid, vs, gs, final))

    # Sort descending by final score
    scored.sort(key=lambda x: x[3], reverse=True)

    # ── hydrate from PostgreSQL ────────────────────────────────────────────────
    results: List[HybridRecommendation] = []
    for pid, vs, gs, final in scored:
        paper_row: Optional[Paper] = db.query(Paper).filter(
            Paper.paper_id == pid,
            Paper.user_id  == user_id,
        ).first()
        if not paper_row:
            continue  # paper may have been deleted

        s_topics  = shared_topics_map.get(pid,  [])
        s_methods = shared_methods_map.get(pid, [])

        # Fallback: derive shared items from stored entities if graph had no edge
        if not s_topics:
            cand_topics = set(paper_row.topics or [])
            s_topics = sorted(source_topics & cand_topics)
        if not s_methods:
            cand_methods = set((paper_row.entities or {}).get("methods", []))
            s_methods = sorted(source_methods & cand_methods)

        results.append(HybridRecommendation(
            paper_id      = pid,
            title         = paper_row.title,
            authors       = paper_row.authors or [],
            abstract      = (paper_row.abstract or "")[:400],
            topics        = paper_row.topics or [],
            vector_score  = round(vs, 4),
            graph_score   = round(gs, 4),
            final_score   = round(min(final, 1.0), 4),   # cap at 1.0
            shared_topics  = s_topics,
            shared_methods = s_methods,
            raw_graph_score = raw_graph.get(pid, 0.0),
            raw_chunk_scores = [],  # not persisted to save memory
            relevance_explanation = _build_explanation(vs, gs, s_topics, s_methods),
        ))

    # ── diversity filter ───────────────────────────────────────────────────────
    results = _apply_diversity_filter(results)

    return results[:top_k]