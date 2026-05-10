from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

_driver = None

# ── Similarity config ─────────────────────────────────────────────────────────
# Raised from 0.40 — prevents cross-domain papers from forming spurious edges
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))
SIMILARITY_TOP_K     = int(os.getenv("SIMILARITY_TOP_K",       "5"))


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


def close_driver():
    global _driver
    if _driver:
        _driver.close()
        _driver = None


# ── Helper ────────────────────────────────────────────────────────────────────

def _norm_set(items: List[str]) -> set:
    """Normalize a list of entity strings to a set of canonical forms."""
    from app.entity_normalizer import normalize_entity
    s = {normalize_entity(x) for x in (items or []) if x and x.strip()}
    s.discard("")
    return s


# ── Core graph creation ───────────────────────────────────────────────────────

def create_paper_graph(
    paper_id: str,
    title: str,
    authors: List[str],
    entities: Dict[str, List[str]],
    topics: List[str],
    user_id: int,
) -> None:
    """Create a paper node and all its related entity nodes in Neo4j.
    FIXED: Entities are normalized before writing so graph overlap is meaningful.
    Neo4j node names are stored in canonical (normalized) form.
    """
    from app.entity_normalizer import normalize_entity_list
    driver = get_driver()
    try:
        # Normalize everything before storage
        entities = {k: normalize_entity_list(v or []) for k, v in entities.items()}
        topics   = normalize_entity_list(topics or [])

        with driver.session() as session:
            session.run(
                """
                MERGE (p:Paper {paper_id: $paper_id})
                SET p.title   = $title,
                    p.user_id = $user_id
                """,
                paper_id=paper_id, title=title, user_id=user_id,
            )

            # Authors
            for author in (authors or [])[:5]:
                if author and author.strip():
                    session.run(
                        """
                        MERGE (a:Author {name: $name})
                        WITH a
                        MATCH (p:Paper {paper_id: $paper_id})
                        MERGE (p)-[:AUTHORED_BY]->(a)
                        """,
                        name=author.strip(), paper_id=paper_id,
                    )

            # Entity nodes — stored with normalized names
            entity_rel_map = {
                "methods":  ("Method",  "USES_METHOD"),
                "datasets": ("Dataset", "USES_DATASET"),
                "models":   ("Model",   "USES_MODEL"),
            }
            for entity_type, (label, rel) in entity_rel_map.items():
                for entity_name in entities.get(entity_type, []):
                    if entity_name and entity_name.strip():
                        session.run(
                            f"""
                            MERGE (e:{label} {{name: $name}})
                            WITH e
                            MATCH (p:Paper {{paper_id: $paper_id}})
                            MERGE (p)-[:{rel}]->(e)
                            """,
                            name=entity_name, paper_id=paper_id,
                        )

            # Topic nodes — stored with normalized names
            for topic in topics:
                if topic and topic.strip():
                    session.run(
                        """
                        MERGE (t:Topic {name: $name})
                        WITH t
                        MATCH (p:Paper {paper_id: $paper_id})
                        MERGE (p)-[:HAS_TOPIC]->(t)
                        """,
                        name=topic, paper_id=paper_id,
                    )

        print(f"✅ Created Neo4j entity graph for paper: {paper_id}")

    except Exception as e:
        print(f"❌ Neo4j graph creation failed for {paper_id}: {e}")
        raise


def store_paper_embedding(paper_id: str, embedding: List[float]) -> None:
    driver = get_driver()
    with driver.session() as session:
        session.run(
            "MATCH (p:Paper {paper_id: $paper_id}) SET p.embedding = $embedding",
            paper_id=paper_id, embedding=embedding,
        )


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.clip(np.dot(va, vb) / (na * nb), 0.0, 1.0))


def create_similarity_edges(
    paper_id: str,
    user_id: int,
    new_embedding: List[float],
    threshold: float = SIMILARITY_THRESHOLD,
    top_k: int = SIMILARITY_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Compare new paper against existing papers using the hybrid similarity model.
    FIXED: normalize entity names from Neo4j before set intersection so
    canonical form matches work (e.g. "RAG" == "rag" == "retrieval-augmented generation").
    """
    driver = get_driver()

    with driver.session() as session:
        records = session.run(
            """
            MATCH (p:Paper {user_id: $user_id})
            WHERE p.paper_id <> $paper_id
              AND p.embedding IS NOT NULL
            OPTIONAL MATCH (p)-[:HAS_TOPIC]->(t:Topic)
            OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Method)
            OPTIONAL MATCH (p)-[:USES_DATASET]->(d:Dataset)
            RETURN p.paper_id               AS pid,
                   p.embedding              AS emb,
                   collect(DISTINCT t.name) AS topics,
                   collect(DISTINCT m.name) AS methods,
                   collect(DISTINCT d.name) AS datasets
            """,
            user_id=user_id, paper_id=paper_id,
        ).data()

    if not records:
        print(f"ℹ️  No existing papers with embeddings for user {user_id}.")
        return []

    with driver.session() as session:
        current = session.run(
            """
            MATCH (p:Paper {paper_id: $paper_id})
            OPTIONAL MATCH (p)-[:HAS_TOPIC]->(t:Topic)
            OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Method)
            OPTIONAL MATCH (p)-[:USES_DATASET]->(d:Dataset)
            RETURN collect(DISTINCT t.name) AS topics,
                   collect(DISTINCT m.name) AS methods,
                   collect(DISTINCT d.name) AS datasets
            """,
            paper_id=paper_id,
        ).single()

    current_topics   = list(current["topics"]   or []) if current else []
    current_methods  = list(current["methods"]  or []) if current else []
    current_datasets = list(current["datasets"] or []) if current else []

    # FIXED: normalize before building paper dict — entities in Neo4j may have
    # been stored by older code before normalization was applied
    current_paper_dict = {
        "paper_id": paper_id,
        "entities": {
            "methods":  list(_norm_set(current_methods)),
            "datasets": list(_norm_set(current_datasets)),
        },
        "topics": list(_norm_set(current_topics)),
    }

    # Normalized sets for intersection (FIXED — was raw set() before)
    cur_topics_norm   = _norm_set(current_topics)
    cur_methods_norm  = _norm_set(current_methods)
    cur_datasets_norm = _norm_set(current_datasets)

    scored: List[Dict[str, Any]] = []
    for record in records:
        emb = record["emb"]
        if emb is None:
            continue

        # FIXED: normalize other paper's entities too before intersection
        other_topics_norm   = _norm_set(list(record.get("topics")   or []))
        other_methods_norm  = _norm_set(list(record.get("methods")  or []))
        other_datasets_norm = _norm_set(list(record.get("datasets") or []))

        other_dict = {
            "paper_id": record["pid"],
            "entities": {
                "methods":  list(other_methods_norm),
                "datasets": list(other_datasets_norm),
            },
            "topics": list(other_topics_norm),
        }

        try:
            from app.similarity_model import score_pair
            result      = score_pair(current_paper_dict, other_dict, new_embedding, emb)
            score       = result.similarity_score
            explanation = result.explanation
        except Exception as e:
            print(f"⚠️ Hybrid similarity fallback for {paper_id[:8]}: {e}")
            score       = _cosine_similarity(new_embedding, emb)
            explanation = ""

        if score < threshold:
            continue

        # FIXED: use normalized intersection for shared entity labels
        shared_topics   = sorted(cur_topics_norm   & other_topics_norm)
        shared_methods  = sorted(cur_methods_norm  & other_methods_norm)
        shared_datasets = sorted(cur_datasets_norm & other_datasets_norm)

        scored.append({
            "paper_id":       record["pid"],
            "score":          round(score, 4),
            "shared_topics":  shared_topics,
            "shared_methods": shared_methods,
            "shared_datasets": shared_datasets,
            "explanation":    explanation,
        })

    if not scored:
        print(f"ℹ️  No papers above threshold {threshold} for {paper_id}.")
        return []

    scored.sort(key=lambda x: x["score"], reverse=True)
    top_matches = scored[:top_k]

    with driver.session() as session:
        for match in top_matches:
            other_id = match["paper_id"]
            session.run(
                """
                MATCH (p1:Paper {paper_id: $id1})
                MATCH (p2:Paper {paper_id: $id2})
                MERGE (p1)-[r:SIMILAR_TO]->(p2)
                  SET r.score           = $score,
                      r.shared_topics   = $shared_topics,
                      r.shared_methods  = $shared_methods,
                      r.shared_datasets = $shared_datasets,
                      r.explanation     = $explanation
                MERGE (p2)-[r2:SIMILAR_TO]->(p1)
                  SET r2.score           = $score,
                      r2.shared_topics   = $shared_topics,
                      r2.shared_methods  = $shared_methods,
                      r2.shared_datasets = $shared_datasets,
                      r2.explanation     = $explanation
                """,
                id1=other_id,  id2=paper_id,
                score=match["score"],
                shared_topics=match["shared_topics"],
                shared_methods=match["shared_methods"],
                shared_datasets=match["shared_datasets"],
                explanation=match["explanation"],
            )
            shared_str = (
                ", ".join(match["shared_methods"][:2] + match["shared_topics"][:2])
                or "semantic"
            )
            print(
                f"  🔗 SIMILAR_TO: {paper_id[:8]}… ↔ {other_id[:8]}…  "
                f"score={match['score']}  mode={match.get('explanation','')[:30]}  via [{shared_str}]"
            )

    print(f"✅ Created {len(top_matches)} SIMILAR_TO edge(s) for paper {paper_id}.")
    return top_matches


# ── Graph retrieval ───────────────────────────────────────────────────────────

def get_paper_graph(paper_id: str) -> Dict[str, Any]:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Paper {paper_id: $paper_id})-[r]->(n)
            RETURN p, type(r) AS rel_type, properties(r) AS rel_props,
                   n, labels(n) AS node_labels
            """,
            paper_id=paper_id,
        )
        nodes = [{"id": paper_id, "label": "Paper", "name": ""}]
        edges = []
        seen  = {paper_id}
        for record in result:
            node      = record["n"]
            rel_type  = record["rel_type"]
            rel_props = record["rel_props"] or {}
            node_id   = str(node.id)
            node_name = node.get("name", node.get("title", "Unknown"))
            if node_id not in seen:
                seen.add(node_id)
                nodes.append({
                    "id":    node_id,
                    "label": (record["node_labels"] or ["Unknown"])[0],
                    "name":  node_name,
                })
            edge = {"source": paper_id, "target": node_id, "type": rel_type}
            if "score"       in rel_props: edge["score"]       = rel_props["score"]
            if "explanation" in rel_props: edge["explanation"] = rel_props["explanation"]
            edges.append(edge)
    return {"nodes": nodes, "edges": edges}


def get_user_graph(user_id: int) -> Dict[str, Any]:
    driver = get_driver()
    with driver.session() as session:
        entity_result = session.run(
            """
            MATCH (p:Paper {user_id: $user_id})-[r]->(n)
            WHERE NOT n:Paper
            RETURN p, type(r) AS rel_type, n, labels(n) AS node_labels
            LIMIT 500
            """,
            user_id=user_id,
        )
        nodes_map: Dict[str, Any] = {}
        edges: List[Dict[str, Any]] = []

        for record in entity_result:
            paper     = record["p"]
            node      = record["n"]
            rel_type  = record["rel_type"]
            node_labels = record["node_labels"]
            paper_key = paper["paper_id"]
            if paper_key not in nodes_map:
                nodes_map[paper_key] = {
                    "id":            paper_key,
                    "label":         "Paper",
                    "name":          paper.get("title", "Unknown Paper"),
                    "cluster_id":    paper.get("cluster_id"),
                    "cluster_label": paper.get("cluster_label", ""),
                }
            node_key = str(node.id)
            if node_key not in nodes_map:
                nodes_map[node_key] = {
                    "id":    node_key,
                    "label": node_labels[0] if node_labels else "Unknown",
                    "name":  node.get("name", "Unknown"),
                }
            edges.append({"source": paper_key, "target": node_key, "type": rel_type})

        sim_result = session.run(
            """
            MATCH (p1:Paper {user_id: $user_id})-[r:SIMILAR_TO]->(p2:Paper {user_id: $user_id})
            RETURN p1.paper_id AS id1, p2.paper_id AS id2,
                   p1.title AS t1, p2.title AS t2,
                   r.score           AS score,
                   r.shared_topics   AS shared_topics,
                   r.shared_methods  AS shared_methods,
                   r.explanation     AS explanation,
                   p1.cluster_id     AS cluster_id_1,
                   p1.cluster_label  AS cluster_label_1,
                   p2.cluster_id     AS cluster_id_2,
                   p2.cluster_label  AS cluster_label_2
            """,
            user_id=user_id,
        )
        for record in sim_result:
            id1, id2 = record["id1"], record["id2"]
            if id1 not in nodes_map:
                nodes_map[id1] = {
                    "id": id1, "label": "Paper", "name": record["t1"] or id1,
                    "cluster_id": record["cluster_id_1"],
                    "cluster_label": record["cluster_label_1"] or "",
                }
            if id2 not in nodes_map:
                nodes_map[id2] = {
                    "id": id2, "label": "Paper", "name": record["t2"] or id2,
                    "cluster_id": record["cluster_id_2"],
                    "cluster_label": record["cluster_label_2"] or "",
                }
            shared_topics  = record["shared_topics"]  or []
            shared_methods = record["shared_methods"] or []
            reason_parts   = []
            if shared_topics:  reason_parts.append(f"topics: {', '.join(shared_topics[:3])}")
            if shared_methods: reason_parts.append(f"methods: {', '.join(shared_methods[:2])}")
            edges.append({
                "source":         id1,
                "target":         id2,
                "type":           "SIMILAR_TO",
                "score":          record["score"],
                "shared_topics":  shared_topics,
                "shared_methods": shared_methods,
                "explanation":    record["explanation"] or "",
                "reason":         "; ".join(reason_parts) if reason_parts else "semantic similarity",
            })

    return {"nodes": list(nodes_map.values()), "edges": edges}


# ── Recommendations ───────────────────────────────────────────────────────────

def get_graph_scores_bulk(paper_id: str, user_id: int) -> Dict[str, Any]:
    driver = get_driver()
    scores:         Dict[str, float]      = {}
    shared_topics:  Dict[str, List[str]]  = {}
    shared_methods: Dict[str, List[str]]  = {}
    with driver.session() as session:
        records = session.run(
            """
            MATCH (src:Paper {paper_id: $paper_id})-[r:SIMILAR_TO]->(tgt:Paper {user_id: $user_id})
            WHERE tgt.paper_id <> $paper_id
            RETURN tgt.paper_id  AS pid, r.score AS score,
                   r.shared_topics  AS st, r.shared_methods AS sm
            """,
            paper_id=paper_id, user_id=user_id,
        ).data()
    for rec in records:
        pid = rec["pid"]
        if pid:
            scores[pid]         = float(rec["score"] or 0.0)
            shared_topics[pid]  = rec["st"] or []
            shared_methods[pid] = rec["sm"] or []
    return {"scores": scores, "shared_topics": shared_topics, "shared_methods": shared_methods}


def find_related_papers(paper_id: str, user_id: int, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Find papers related to `paper_id` using a two-signal approach:
      1. Shared entity nodes (Method / Dataset / Topic) via graph traversal.
      2. SIMILAR_TO edge score from the hybrid similarity model.

    BUG FIXES vs original:
    ──────────────────────
    FIX 1 — SIMILAR_TO direction was reversed in the OPTIONAL MATCH.
      The edge is stored as (other)-[:SIMILAR_TO]->(source) because
      create_similarity_edges() writes `MERGE (p1)-[r:SIMILAR_TO]->(p2)`
      with `id1=other_id, id2=paper_id`. So the correct traversal is
      either direction (undirected), not (source)-[:SIMILAR_TO]->(related).
      Old query:  OPTIONAL MATCH (src:Paper {paper_id})-[sim:SIMILAR_TO]->(related)
      New query:  OPTIONAL MATCH (src:Paper {paper_id})-[sim:SIMILAR_TO]-(related)
      This was causing sim.score to always be null → coalesce → 0.00 on the
      recommendations page.

    FIX 2 — Papers with only semantic similarity (no shared entity nodes)
      were dropped from results. When entity_overlap=0 AND sim_score=0
      (due to FIX 1 not being applied), combined_score=0 for all
      semantically-similar papers, so ORDER BY DESC still put entity papers
      first but semantic-only papers disappeared. The chocolate papers had no
      shared normalized entity nodes so they never appeared for each other.
      New query uses UNION to merge entity-overlap results with SIMILAR_TO
      edge results, then deduplicates and re-ranks by combined_score.

    FIX 3 — combined_score formula had a `* 10` multiplier on sim_score
      that made no mathematical sense (sim_score is 0–1, entity_overlap is
      0–N, but weighting 10× sim_score dominated when it was non-zero).
      New formula: (entity_overlap_norm * 0.5) + (sim_score * 0.5)
      where entity_overlap_norm = entity_overlap / (entity_overlap + 5)
      to softly cap the entity signal contribution.
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            // ── Branch A: papers with shared entity nodes ─────────────────────
            MATCH (src:Paper {paper_id: $paper_id})
            OPTIONAL MATCH (src)-[]->(shared)<-[]-(related:Paper {user_id: $user_id})
            WHERE related.paper_id <> $paper_id
              AND NOT shared:Author
              AND NOT shared:Paper
            WITH src, related,
                 count(DISTINCT shared)          AS entity_overlap,
                 collect(DISTINCT shared.name)[..5] AS shared_items
            // FIX 1: undirected SIMILAR_TO so both edge directions are matched
            OPTIONAL MATCH (src)-[sim:SIMILAR_TO]-(related)
            WITH related,
                 coalesce(entity_overlap, 0)     AS entity_overlap,
                 coalesce(shared_items,   [])    AS shared_items,
                 coalesce(sim.score,      0.0)   AS sim_score
            WHERE related IS NOT NULL

            // FIX 3: balanced combined score — no artificial *10 multiplier
            WITH related, entity_overlap, shared_items, sim_score,
                 // Soft-cap entity overlap contribution: 5 overlaps ≈ 0.5 weight
                 (toFloat(entity_overlap) / (toFloat(entity_overlap) + 5.0) * 0.5
                  + sim_score * 0.5)             AS combined_score
            WHERE related.paper_id IS NOT NULL

            RETURN related.paper_id              AS paper_id,
                   related.title                 AS title,
                   entity_overlap,
                   shared_items,
                   sim_score,
                   combined_score

            UNION

            // ── Branch B: papers linked only via SIMILAR_TO (FIX 2) ──────────
            // Catches papers that share no entity nodes but have high embedding
            // similarity (e.g. two chocolate papers with different topic tags).
            MATCH (src:Paper {paper_id: $paper_id})
            // FIX 1: undirected match
            MATCH (src)-[sim:SIMILAR_TO]-(related:Paper {user_id: $user_id})
            WHERE related.paper_id <> $paper_id
            RETURN related.paper_id              AS paper_id,
                   related.title                 AS title,
                   0                             AS entity_overlap,
                   []                            AS shared_items,
                   sim.score                     AS sim_score,
                   (sim.score * 0.5)             AS combined_score
            """,
            paper_id=paper_id, user_id=user_id,
        ).data()

    # Deduplicate by paper_id — keep the highest combined_score per paper
    seen: dict = {}
    for r in result:
        pid = r.get("paper_id")
        if not pid:
            continue
        if pid not in seen or r["combined_score"] > seen[pid]["combined_score"]:
            seen[pid] = r

    ranked = sorted(seen.values(), key=lambda x: x["combined_score"], reverse=True)[:limit]

    return [
        {
            "paper_id":         r["paper_id"],
            "title":            r["title"],
            "shared_count":     r["entity_overlap"],
            "shared_items":     r["shared_items"],
            # FIX 1: sim_score is now correctly read from the edge
            "similarity_score": round(float(r["sim_score"] or 0), 4),
            "combined_score":   round(float(r["combined_score"] or 0), 4),
        }
        for r in ranked
    ]


# ── Cleanup ───────────────────────────────────────────────────────────────────

def delete_paper_nodes(paper_id: str):
    driver = get_driver()
    with driver.session() as session:
        session.run(
            "MATCH (p:Paper {paper_id: $paper_id}) DETACH DELETE p",
            paper_id=paper_id,
        )
    print(f"✅ Deleted Neo4j nodes for paper: {paper_id}")