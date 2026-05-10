"""
graph_context.py — FIXED v2
────────────────────────────────────────────────────────────────────────────────
Graph Context Builder for GraphRAG reasoning.

Fixes vs v1
───────────
1. All entities normalized via normalize_entity_list() BEFORE any set
   intersection — was missing in original, causing "0% normalization hit rate".
2. "has_signal" is only True when REAL overlap exists (not inferred from text).
3. Weak-signal fallback: when no direct overlap, top-3 shared topics by Jaccard
   are added as "weak_signal_topics" and marked explicitly.
4. similarity_signals now include numeric Jaccard scores for eval pipeline.
5. graph_context_to_str() now emits "graph_reason" field used by LLM prompt.

Public API (unchanged + weak_signal_topics added to GraphContext)
──────────
  build_graph_context(papers: list[dict]) -> GraphContext
  graph_context_to_str(ctx: GraphContext) -> str
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GraphContext:
    shared_methods:      List[str]            = field(default_factory=list)
    shared_datasets:     List[str]            = field(default_factory=list)
    shared_topics:       List[str]            = field(default_factory=list)
    shared_models:       List[str]            = field(default_factory=list)
    unique_methods:      Dict[str, List[str]] = field(default_factory=dict)
    unique_datasets:     Dict[str, List[str]] = field(default_factory=dict)
    connections:         List[str]            = field(default_factory=list)
    similarity_signals:  List[Dict]           = field(default_factory=list)
    weak_signal_topics:  List[str]            = field(default_factory=list)  # NEW
    has_signal:          bool                 = False
    signal_type:         str                  = "none"  # "strong" | "weak" | "none"

    def to_dict(self) -> dict:
        return {
            "shared_methods":    self.shared_methods,
            "shared_datasets":   self.shared_datasets,
            "shared_topics":     self.shared_topics,
            "shared_models":     self.shared_models,
            "unique_methods":    self.unique_methods,
            "unique_datasets":   self.unique_datasets,
            "connections":       self.connections,
            "similarity_signals": self.similarity_signals,
            "weak_signal_topics": self.weak_signal_topics,
            "has_signal":        self.has_signal,
            "signal_type":       self.signal_type,
        }


# ── Core builder ──────────────────────────────────────────────────────────────

def build_graph_context(
    papers: List[dict],
    top_k: int = 3,
) -> GraphContext:
    """
    Build a structured graph context from N papers.

    FIXED: All entities are normalized before set intersection, so canonical
    forms (e.g. "rag") match across papers that used different surface forms.

    Each paper dict should contain:
        paper_id : str
        title    : str
        entities : dict  with keys methods / datasets / models / tasks / topics
        topics   : list[str]
    """
    from app.entity_normalizer import normalize_entity_list, normalize_entity

    if len(papers) < 2:
        return GraphContext()

    # ── Normalize all entities FIRST ──────────────────────────────────────────
    normalized_papers: List[dict] = []
    for p in papers:
        ents = p.get("entities") or {}
        np_ = {
            "paper_id": p.get("paper_id", ""),
            "title":    _short_title(p.get("title", "")),
            # FIXED: normalize_entity_list applied to every field
            "methods":  set(normalize_entity_list(ents.get("methods",  []) or [])),
            "datasets": set(normalize_entity_list(ents.get("datasets", []) or [])),
            "models":   set(normalize_entity_list(ents.get("models",   []) or [])),
            "tasks":    set(normalize_entity_list(ents.get("tasks",    []) or [])),
            "topics":   set(normalize_entity_list(p.get("topics",      []) or [])),
        }
        # Remove empty strings
        for k in ("methods", "datasets", "models", "tasks", "topics"):
            np_[k].discard("")
        normalized_papers.append(np_)

    # ── Shared across ALL papers ───────────────────────────────────────────────
    def _all_shared(fld: str) -> List[str]:
        if not normalized_papers:
            return []
        common: Set[str] = normalized_papers[0][fld].copy()
        for p in normalized_papers[1:]:
            common &= p[fld]
        return sorted(common)[:top_k]

    shared_methods  = _all_shared("methods")
    shared_datasets = _all_shared("datasets")
    shared_models   = _all_shared("models")
    shared_topics   = _all_shared("topics")

    # ── Unique entities per paper ─────────────────────────────────────────────
    unique_methods:  Dict[str, List[str]] = {}
    unique_datasets: Dict[str, List[str]] = {}
    for p in normalized_papers:
        other_methods  = set().union(*(q["methods"]  for q in normalized_papers if q is not p))
        other_datasets = set().union(*(q["datasets"] for q in normalized_papers if q is not p))
        um = sorted(p["methods"]  - other_methods)[:top_k]
        ud = sorted(p["datasets"] - other_datasets)[:top_k]
        if um:
            unique_methods[p["paper_id"]]  = um
        if ud:
            unique_datasets[p["paper_id"]] = ud

    # ── Pair-wise connections with Jaccard scores ──────────────────────────────
    connections:        List[str] = []
    similarity_signals: List[dict] = []
    all_pair_topics:    List[Tuple[str, float]] = []  # for weak-signal fallback

    n = len(normalized_papers)
    for i in range(n):
        for j in range(i + 1, n):
            pa = normalized_papers[i]
            pb = normalized_papers[j]
            label_a = pa["title"]
            label_b = pb["title"]

            # Per-pair Jaccard (already normalized above)
            pair_methods  = sorted(pa["methods"]  & pb["methods"])[:top_k]
            pair_datasets = sorted(pa["datasets"] & pb["datasets"])[:top_k]
            pair_topics   = sorted(pa["topics"]   & pb["topics"])[:top_k]
            pair_models   = sorted(pa["models"]   & pb["models"])[:top_k]

            def _jaccard_score(a: set, b: set) -> float:
                union = a | b
                if not union:
                    return 0.0
                return len(a & b) / len(union)

            m_score = _jaccard_score(pa["methods"],  pb["methods"])
            d_score = _jaccard_score(pa["datasets"], pb["datasets"])
            t_score = _jaccard_score(pa["topics"],   pb["topics"])

            # Track topics for weak-signal fallback
            if t_score > 0:
                for t in pair_topics:
                    all_pair_topics.append((t, t_score))

            # Build connection strings
            if pair_methods:
                connections.append(
                    f'"{label_a}" and "{label_b}" share method(s): {", ".join(pair_methods)}'
                )
            if pair_datasets:
                connections.append(
                    f'"{label_a}" and "{label_b}" share dataset(s): {", ".join(pair_datasets)}'
                )
            if pair_topics:
                connections.append(
                    f'"{label_a}" and "{label_b}" share topic(s): {", ".join(pair_topics)}'
                )
            if pair_models and not pair_methods:
                connections.append(
                    f'"{label_a}" and "{label_b}" share model(s): {", ".join(pair_models)}'
                )

            dominant = (
                "methods"  if pair_methods  else
                "datasets" if pair_datasets else
                "topics"   if pair_topics   else
                "none"
            )

            similarity_signals.append({
                "pair":            f"{label_a} ↔ {label_b}",
                "pair_ids":        (pa["paper_id"], pb["paper_id"]),
                "shared_methods":  pair_methods,
                "shared_datasets": pair_datasets,
                "shared_topics":   pair_topics,
                "dominant":        dominant,
                "method_score":    round(m_score, 4),
                "dataset_score":   round(d_score, 4),
                "topic_score":     round(t_score, 4),
            })

            if dominant == "none":
                connections.append(
                    f'"{label_a}" and "{label_b}" have no shared entities — '
                    f"similarity is semantic only"
                )

    # ── Determine signal type ─────────────────────────────────────────────────
    has_strong_signal = bool(
        shared_methods or shared_datasets or shared_topics or shared_models or
        any(s["dominant"] != "none" for s in similarity_signals)
    )

    # FIXED: Weak-signal fallback — use top topic Jaccard if no direct overlap
    weak_signal_topics: List[str] = []
    signal_type = "none"

    if has_strong_signal:
        signal_type = "strong"
    elif all_pair_topics:
        # Sort by Jaccard score and take top-3
        seen: Set[str] = set()
        for topic, score in sorted(all_pair_topics, key=lambda x: -x[1]):
            if topic not in seen and score >= 0.3:
                weak_signal_topics.append(topic)
                seen.add(topic)
            if len(weak_signal_topics) >= top_k:
                break
        if weak_signal_topics:
            signal_type = "weak"
            connections.append(
                f"Weak graph signal: papers share related topics — "
                f"{', '.join(weak_signal_topics)}"
            )

    has_signal = signal_type != "none"

    return GraphContext(
        shared_methods    = shared_methods,
        shared_datasets   = shared_datasets,
        shared_topics     = shared_topics,
        shared_models     = shared_models,
        unique_methods    = unique_methods,
        unique_datasets   = unique_datasets,
        connections       = connections[:9],
        similarity_signals = similarity_signals,
        weak_signal_topics = weak_signal_topics,
        has_signal        = has_signal,
        signal_type       = signal_type,
    )


def graph_context_to_str(ctx: GraphContext) -> str:
    """
    Serialize GraphContext to a compact string for LLM prompt injection.

    FIXED: Emits graph_reason field so LLM prompt can reference it directly.
    Adds explicit "weakly related" instruction when no strong signal exists.
    Budget: ~300–500 tokens.
    """
    lines: List[str] = ["=== GRAPH CONTEXT (entity overlap — treat as ground truth) ==="]

    if ctx.signal_type == "none":
        lines.append("No shared entities detected across papers.")
        lines.append("⚠ STRICT: State papers are WEAKLY RELATED. Do NOT infer similarity from text alone.")
        lines.append('graph_reason: "no shared entities — semantic similarity only"')
        return "\n".join(lines)

    if ctx.signal_type == "weak":
        lines.append("⚠ WEAK GRAPH SIGNAL: No direct method/dataset overlap.")
        if ctx.weak_signal_topics:
            lines.append(f"Related topics (inferred): {', '.join(ctx.weak_signal_topics)}")
        lines.append('graph_reason: "weak topic overlap — weakly related"')
        lines.append("State explicitly that relationship is weak.")
    else:
        # Strong signal
        if ctx.shared_methods:
            lines.append(f"Shared methods (ALL papers): {', '.join(ctx.shared_methods)}")
        if ctx.shared_datasets:
            lines.append(f"Shared datasets (ALL papers): {', '.join(ctx.shared_datasets)}")
        if ctx.shared_topics:
            lines.append(f"Shared topics (ALL papers): {', '.join(ctx.shared_topics)}")
        if ctx.shared_models:
            lines.append(f"Shared models (ALL papers): {', '.join(ctx.shared_models)}")

        # Graph reason
        primary = (
            ctx.shared_methods  or
            ctx.shared_datasets or
            ctx.shared_topics   or
            ["semantic similarity"]
        )
        lines.append(f'graph_reason: "shared {("methods" if ctx.shared_methods else "datasets" if ctx.shared_datasets else "topics")}: {", ".join(primary[:3])}"')

    # Unique methods
    for pid, methods in ctx.unique_methods.items():
        lines.append(f"Unique to {pid[:8]}…: {', '.join(methods)}")

    # Connections
    if ctx.connections:
        lines.append("Connections:")
        for conn in ctx.connections:
            lines.append(f"  • {conn}")

    return "\n".join(lines)


# ── Helper ────────────────────────────────────────────────────────────────────

def _short_title(title: str, max_len: int = 35) -> str:
    if not title:
        return "Unknown"
    return title[:max_len] + "…" if len(title) > max_len else title