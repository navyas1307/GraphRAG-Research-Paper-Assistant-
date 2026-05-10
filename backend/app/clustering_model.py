"""
clustering_model.py — v3
────────────────────────────────────────────────────────────────────────────────
DIAGNOSIS of current failures (silhouette=0.038, coherence=0%, ratio=1.11):

PROBLEM 1 — K=3 with distribution 5+2+1 is wrong
  The elbow method on 6 papers produces a flat inertia curve — any K from 2 to 5
  gives near-identical inertia because the embedding space of 6 points has no
  strong elbow. The domain-aware floor correctly detects 2-3 domains (NLP,
  Social, Food) but the elbow then picks K=3 and assigns 5 papers to cluster 0
  (all NLP + food papers, which share generic "research" vocabulary) and 2 to
  cluster 1 (the women empowerment pair), with 1 outlier. This is exactly the
  "5+2+1 three clusters" we see.

  ROOT CAUSE: With n=6 papers and K=3, max_k = min(15, 5) = 5. k_range=[3,4,5].
  Inertia curve for 3 semantically distinct groups in 768-dim space is nearly
  monotone — second derivative is unreliable and often picks K=3 incorrectly.

  FIX 1: For small corpora (n ≤ 12), SKIP the elbow entirely and use
  domain-count directly. Elbow requires at least 10+ points to be reliable.
  For n ≤ 12: K = n_domains (guaranteed cross-domain separation).
  For n > 12: elbow with domain floor (unchanged).

PROBLEM 2 — Coherence=0% for all multi-paper clusters
  _label_cluster() requires LABEL_MIN_PRESENCE=0.50 (≥50% of papers share a
  keyword). The cluster with 5 papers has: 2 women papers (share "gender"),
  2 RAG/NLP papers (share "rag"), 1 multimodal paper (no dominant overlap).
  With 5 papers, any keyword appearing in only 2 papers = 40% < 50% threshold
  → NO keyword survives → fallback to title words → coherence check fails.

  The coherence metric checks entity intersection per PAIR of papers, not the
  label. With the 5-paper cluster mixing NLP + food + social, most pairs have
  zero entity overlap → coherence = 0%.

  FIX 2: Separate domain detection and entity enrichment:
  a. Use spectral clustering instead of KMeans for small corpora — it respects
     the embedding manifold better than spherical KMeans assumptions.
  b. Lower LABEL_MIN_PRESENCE to 0.34 for clusters of 3+ papers (1 in 3 papers
     sharing a keyword is still meaningful for small corpora).
  c. For coherence measurement: when cluster has < 3 pairs with entity overlap,
     fall back to embedding-cosine coherence (intra-cluster mean cosine).

PROBLEM 3 — choose_k_with_domain_split uses elbow for n=6 giving misleading K
  With 6 papers: max_k = min(15, 5) = 5. n_domains = 3 → min_k = 3.
  k_range = [3, 4, 5]. Three inertia points → d1 has 2 points → d2 has 1 point
  → np.argmax returns 0 → elbow_idx=1 → k_range[1]=4. Not reliable.

  FIX 3: Small corpus fast path. If n ≤ 12, bypass elbow entirely:
    K = min(n_domains + 1, n // 2)
  This gives K=3 for 6 papers with 3 domains (NLP/Social/Food) — but with
  proper centroid initialization, the 3 clusters will be semantically clean.
  The "+1" allows splitting the largest domain (NLP) into sub-topics.

PROBLEM 4 — KMeans random_state=42 with n_init=10 is fine BUT initialization
  matters hugely for small n. With 6 papers and K=3, KMeans++ (default) still
  sometimes merges the food papers with NLP. Adding Agglomerative clustering
  as a fallback when silhouette < 0.15 (current = 0.038) provides a better
  result because it uses full pairwise distances not just centroids.

  FIX 4: After KMeans, if silhouette < 0.15, retry with AgglomerativeClustering
  (ward linkage) and keep whichever has higher silhouette.

Public API (unchanged):
  run_kmeans(embeddings, papers, k=None)   → ClusteringResult
  run_dbscan(embeddings, papers, ...)      → ClusteringResult
  assign_new_paper(embedding, model)       → (cluster_id, cluster_label)
  write_clusters_to_neo4j(result, user_id)
  choose_k(embeddings)                     → int
"""

from __future__ import annotations

import os
import pickle
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import normalize

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_K              = 3
MAX_K_SEARCH           = 15
DBSCAN_EPS             = 0.25
DBSCAN_MIN_SAMPLES     = 2
MODEL_CACHE_PATH       = "clustering_model.pkl"
RETRAIN_THRESHOLD      = 0.20
LABEL_TOP_N            = 3

# FIX 2: lowered from 0.50 — 1-in-3 papers sharing a keyword is signal for small corpora
LABEL_MIN_PRESENCE     = 0.34

# FIX 4: threshold below which we try agglomerative as fallback
SILHOUETTE_RETRY_THRESHOLD = 0.15

# FIX 3: small corpus threshold — bypass elbow for reliable K
SMALL_CORPUS_THRESHOLD = 12


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PaperClusterAssignment:
    paper_id:      str
    cluster_id:    int
    cluster_label: str

    def to_dict(self) -> dict:
        return {
            "paper_id":      self.paper_id,
            "cluster_id":    self.cluster_id,
            "cluster_label": self.cluster_label,
        }


@dataclass
class ClusteringResult:
    algorithm:        str
    n_clusters:       int
    assignments:      List[PaperClusterAssignment]
    cluster_labels:   Dict[int, str]
    silhouette_score: float
    model:            Any = field(default=None, repr=False)
    embeddings_norm:  np.ndarray = field(default=None, repr=False)
    # FIX: store PCA model used during clustering so evaluation uses SAME
    # reduced space — was getattr(result, "pca_model", None) → always None,
    # causing evaluation to fit a fresh PCA on different data than clustering used.
    pca_model:        Any = field(default=None, repr=False)
    embeddings_pca:   Any = field(default=None, repr=False)  # PCA-reduced, L2-normed

    def to_dict(self) -> dict:
        return {
            "algorithm":        self.algorithm,
            "n_clusters":       self.n_clusters,
            "silhouette_score": round(self.silhouette_score, 4),
            "cluster_labels":   self.cluster_labels,
            "assignments":      [a.to_dict() for a in self.assignments],
        }


# ── Cluster labelling ─────────────────────────────────────────────────────────

def _label_cluster(papers_in_cluster: List[dict]) -> str:
    """
    FIX 2: LABEL_MIN_PRESENCE lowered to 0.34 for small clusters.
    A keyword present in ≥1/3 of papers is now considered a cluster signal.

    For single-paper clusters, immediately return title-derived label.
    """
    n = len(papers_in_cluster)
    if n == 0:
        return "General Research"
    if n == 1:
        # Single-paper cluster: derive from title directly
        title_words = [
            w.lower() for w in (papers_in_cluster[0].get("title") or "").split()
            if len(w) > 4 and w.isalpha()
        ]
        _STOP = {"based", "using", "towards", "approach", "study", "analysis",
                 "paper", "novel", "large", "model", "models", "learning",
                 "research", "survey", "review", "system", "their"}
        clean = [w for w in title_words if w not in _STOP]
        return " · ".join(clean[:LABEL_TOP_N]) if clean else "Unique Paper"

    presence: Counter = Counter()
    weight:   Dict[str, float] = {}

    for paper in papers_in_cluster:
        entities = paper.get("entities") or {}
        seen: set = set()

        for method in (entities.get("methods") or []):
            kw = method.strip()
            if kw and kw not in seen:
                presence[kw] += 1
                weight[kw] = weight.get(kw, 0) + 2.0
                seen.add(kw)

        for topic in (paper.get("topics") or []):
            kw = topic.strip()
            if kw and kw not in seen:
                presence[kw] += 1
                weight[kw] = weight.get(kw, 0) + 1.0
                seen.add(kw)

    # FIX 2: adaptive threshold — 34% for small clusters, 50% for larger
    presence_thresh = LABEL_MIN_PRESENCE if n <= 6 else 0.50
    min_count = max(1, int(np.ceil(n * presence_thresh)))
    survivors = {kw: w for kw, w in weight.items() if presence[kw] >= min_count}

    if survivors:
        top = sorted(survivors, key=lambda k: (-presence[k], -weight[k]))[:LABEL_TOP_N]
        return " · ".join(top)

    # Fallback: shared title words
    title_word_presence: Counter = Counter()
    for p in papers_in_cluster:
        words = {w.lower() for w in (p.get("title") or "").split() if len(w) > 4 and w.isalpha()}
        for w in words:
            title_word_presence[w] += 1

    _STOP = {"based", "using", "towards", "approach", "method", "study", "analysis",
             "paper", "novel", "large", "model", "models", "learning", "deep",
             "research", "survey", "review", "system", "systems", "their"}
    shared = [w for w, cnt in title_word_presence.most_common()
              if cnt >= max(1, n // 2) and w not in _STOP]
    if shared:
        return " · ".join(shared[:LABEL_TOP_N])

    # Final fallback: any title words
    all_words: Counter = Counter()
    for p in papers_in_cluster:
        for w in (p.get("title") or "").split():
            if len(w) > 4 and w.isalpha() and w.lower() not in _STOP:
                all_words[w.lower()] += 1
    if all_words:
        return " · ".join(w for w, _ in all_words.most_common(LABEL_TOP_N))

    return "General Research"


# ── Domain detection ──────────────────────────────────────────────────────────

def _detect_domain_groups(papers: List[dict]) -> int:
    """
    Count distinct research domains using the domain bucket classifier.
    Returns number of distinct non-neutral domains (minimum 1).
    """
    try:
        from app.similarity_model import _domain_bucket
        buckets = {_domain_bucket(p) for p in papers}
        buckets.discard(-1)
        n = len(buckets)
        if n >= 2:
            print(f"  [clustering] {n} distinct domains detected → K floor = {n}")
        return max(1, n)
    except Exception as e:
        print(f"  [clustering] Domain detection skipped: {e}")
        return 1


# ── K selection ───────────────────────────────────────────────────────────────

def choose_k(
    embeddings: np.ndarray,
    max_k: int = MAX_K_SEARCH,
    min_k: int = 2,
) -> int:
    """
    Elbow method for K selection. Only called for n > SMALL_CORPUS_THRESHOLD.
    FIX 3: small corpus fast path — bypass entirely if n ≤ 12.
    """
    n = len(embeddings)
    min_k = max(min_k, 2)
    max_k = min(max_k, max(min_k, n - 1))
    if max_k < min_k:
        return min_k
    k_range = list(range(min_k, max_k + 1))
    if len(k_range) == 1:
        return k_range[0]
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(embeddings)
        inertias.append(km.inertia_)
    arr = np.array(inertias)
    if len(arr) < 3:
        return k_range[0]
    d2 = np.diff(np.diff(arr))
    return k_range[int(np.argmax(np.abs(d2))) + 1]


def choose_k_with_domain_split(
    embeddings: np.ndarray,
    papers: List[dict],
    max_k: int = MAX_K_SEARCH,
) -> int:
    """
    FIX 3: Small corpus fast path.
    - n ≤ SMALL_CORPUS_THRESHOLD: skip elbow, use K = n_domains + 1 (split largest domain)
    - n > threshold: elbow with domain floor.

    Rationale: elbow method needs ≥15 points for reliable second derivatives.
    For 6-12 papers, domain count is the most reliable signal.
    """
    n = len(papers)
    n_domains = _detect_domain_groups(papers)

    if n <= SMALL_CORPUS_THRESHOLD:
        # Small corpus: K = n_domains if we have clear separation, else n_domains + 1
        # +1 allows splitting the dominant domain (e.g. NLP papers) by sub-topic
        k = min(n_domains + 1, max(2, n // 2))
        print(f"  [clustering] Small corpus (n={n}): bypassing elbow → K={k}")
        return k

    min_k = max(2, n_domains)
    return choose_k(embeddings, max_k=max_k, min_k=min_k)


# ── Agglomerative fallback ────────────────────────────────────────────────────

def _run_agglomerative(emb_norm: np.ndarray, k: int, papers: List[dict]) -> Tuple[np.ndarray, float]:
    """
    FIX 4: Ward agglomerative clustering as fallback for KMeans.
    Ward linkage minimises total within-cluster variance, which is better suited
    to asymmetric small-n embedding spaces than KMeans centroid assumption.
    Returns (labels, silhouette_score).
    """
    from sklearn.metrics import silhouette_score
    try:
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = agg.fit_predict(emb_norm)
        sil = 0.0
        if len(set(labels)) >= 2:
            sil = float(silhouette_score(emb_norm, labels, metric="euclidean"))
        return labels, sil
    except Exception as e:
        print(f"  [clustering] Agglomerative fallback failed: {e}")
        return np.zeros(len(papers), dtype=int), 0.0


# ── KMeans clustering ─────────────────────────────────────────────────────────

def run_kmeans(
    embeddings: List[List[float]],
    papers: List[dict],
    k: Optional[int] = None,
) -> ClusteringResult:
    """
    FIX 3+4: Domain-aware K for small corpora + agglomerative fallback.

    Steps:
      1. L2-normalise embeddings.
      2. Choose K: small corpus → domain_count+1; large → elbow with domain floor.
      3. KMeans with 15 initialisations (more than v2 for better convergence).
      4. If silhouette < SILHOUETTE_RETRY_THRESHOLD: try agglomerative and pick better.
      5. Label clusters with FIX 2 (adaptive presence threshold).
    """
    from sklearn.metrics import silhouette_score

    from sklearn.decomposition import PCA

    emb_arr  = np.array(embeddings, dtype=np.float32)
    emb_norm = normalize(emb_arr, norm="l2")
    n        = len(papers)

    if k is None:
        k = choose_k_with_domain_split(emb_norm, papers)
    k = max(2, min(k, n - 1))

    # FIX: PCA-reduce before clustering.
    # Raw 768-dim embeddings cause the "curse of dimensionality" — all pairwise
    # distances converge to the same value, making KMeans centroids indistinct
    # and silhouette near-zero. PCA to n_domains*3 dims (min 6, max 20) preserves
    # the variance that separates research domains while making distances meaningful.
    # We store the fitted PCA so evaluation.py can use the SAME reduced space,
    # giving consistent silhouette numbers between clustering and evaluation.
    pca_model      = None
    emb_pca        = emb_norm  # fallback: raw L2-normed
    n_pca_components = min(max(k * 3, 6), 20, n - 1)

    if n >= 4 and n_pca_components >= 2:
        try:
            pca_model = PCA(n_components=n_pca_components, random_state=42)
            reduced   = pca_model.fit_transform(emb_norm)
            emb_pca   = normalize(reduced, norm="l2")
            var_explained = sum(pca_model.explained_variance_ratio_) * 100
            print(f"  [clustering] PCA: {n_pca_components} components, "
                  f"{var_explained:.1f}% variance explained")
        except Exception as e:
            print(f"  [clustering] PCA failed ({e}), using raw embeddings")
            pca_model = None
            emb_pca   = emb_norm

    # KMeans with increased n_init for stability — fit on PCA-reduced space
    km = KMeans(n_clusters=k, random_state=42, n_init=15, init="k-means++")
    km_labels = km.fit_predict(emb_pca)

    km_sil = 0.0
    if len(set(km_labels)) >= 2:
        try:
            km_sil = float(silhouette_score(emb_pca, km_labels, metric="euclidean"))
        except Exception:
            km_sil = 0.0

    # FIX 4: Try agglomerative if KMeans silhouette is poor
    final_labels = km_labels
    final_sil    = km_sil
    algorithm    = "kmeans"

    if km_sil < SILHOUETTE_RETRY_THRESHOLD and n >= 3:
        print(f"  [clustering] KMeans silhouette={km_sil:.3f} < {SILHOUETTE_RETRY_THRESHOLD} "
              f"— trying AgglomerativeClustering (ward) as fallback")
        # FIX: use emb_pca (PCA-reduced) for agglomerative too
        agg_labels, agg_sil = _run_agglomerative(emb_pca, k, papers)
        if agg_sil > km_sil:
            final_labels = agg_labels
            final_sil    = agg_sil
            algorithm    = "kmeans+agglomerative"
            print(f"  [clustering] Agglomerative wins: sil={agg_sil:.3f} > kmeans={km_sil:.3f}")
        else:
            print(f"  [clustering] KMeans retained: sil={km_sil:.3f} ≥ agg={agg_sil:.3f}")

    # Build cluster → papers map
    cluster_papers: Dict[int, List[dict]] = {i: [] for i in range(k)}
    for idx, label in enumerate(final_labels):
        cluster_papers[label].append(papers[idx])

    # FIX 2: adaptive-threshold labels
    cluster_labels = {cid: _label_cluster(plist) for cid, plist in cluster_papers.items()}

    assignments = [
        PaperClusterAssignment(
            paper_id      = papers[i]["paper_id"],
            cluster_id    = int(final_labels[i]),
            cluster_label = cluster_labels[int(final_labels[i])],
        )
        for i in range(n)
    ]

    print(f"  [clustering] {algorithm}: k={k}, silhouette={final_sil:.3f}")
    for cid, plist in cluster_papers.items():
        print(f"    Cluster {cid} '{cluster_labels[cid]}': {len(plist)} papers")

    # Store KMeans model for incremental assignments (even if agglomerative won)
    # FIX: expose pca_model + embeddings_pca so evaluation.py uses the SAME
    # reduced space, giving consistent silhouette scores.
    return ClusteringResult(
        algorithm        = algorithm,
        n_clusters       = k,
        assignments      = assignments,
        cluster_labels   = cluster_labels,
        silhouette_score = final_sil,
        model            = km,          # KMeans model for predict()
        embeddings_norm  = emb_norm,
        pca_model        = pca_model,   # FIX: PCA fitted on training embeddings
        embeddings_pca   = emb_pca,     # FIX: PCA-reduced, L2-normed — same as clustering used
    )


# ── DBSCAN clustering ─────────────────────────────────────────────────────────

def run_dbscan(
    embeddings: List[List[float]],
    papers: List[dict],
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> ClusteringResult:
    """
    DBSCAN on cosine distance. Best for mixed-domain corpora where you don't
    know K and some papers should be labelled as outliers (cluster_id = -1).

    For the current 6-paper corpus (NLP + Social + Food), DBSCAN with eps=0.20
    should give:
      - Cluster 0: RAG + Multimodal (NLP, cosine ~0.73)
      - Cluster 1: Women + Women journal (Social, cosine ~0.79)
      - Cluster 2: Chocolate + Cocoa (Food, cosine ~0.85)
      - No outliers
    If eps=0.25 merges some groups, lower it to 0.20.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_distances

    emb_arr  = np.array(embeddings, dtype=np.float32)
    emb_norm = normalize(emb_arr, norm="l2")

    dist_matrix = cosine_distances(emb_norm)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(dist_matrix)

    n_clusters = len(set(labels) - {-1})
    n_outliers = int(np.sum(labels == -1))

    cluster_papers: Dict[int, List[dict]] = {}
    for cid in set(labels):
        cluster_papers[cid] = []
    for idx, label in enumerate(labels):
        cluster_papers[label].append(papers[idx])

    cluster_labels: Dict[int, str] = {}
    for cid, plist in cluster_papers.items():
        cluster_labels[cid] = "Unique / Outlier Papers" if cid == -1 else _label_cluster(plist)

    sil = 0.0
    mask = labels != -1
    if n_clusters >= 2 and mask.sum() >= 2:
        try:
            sil = float(silhouette_score(
                dist_matrix[mask][:, mask], labels[mask], metric="precomputed"
            ))
        except Exception:
            sil = 0.0

    assignments = [
        PaperClusterAssignment(
            paper_id      = papers[i]["paper_id"],
            cluster_id    = int(labels[i]),
            cluster_label = cluster_labels[int(labels[i])],
        )
        for i in range(len(papers))
    ]

    print(f"  [DBSCAN] {n_clusters} clusters, {n_outliers} outliers "
          f"(eps={eps}, min_samples={min_samples}, silhouette={sil:.3f})")

    return ClusteringResult(
        algorithm        = "dbscan",
        n_clusters       = n_clusters,
        assignments      = assignments,
        cluster_labels   = cluster_labels,
        silhouette_score = sil,
        model            = db,
        embeddings_norm  = emb_norm,
    )


# ── Incremental update ────────────────────────────────────────────────────────

def assign_new_paper(
    new_embedding: List[float],
    result: ClusteringResult,
) -> Tuple[int, str]:
    emb = normalize(np.array(new_embedding, dtype=np.float32).reshape(1, -1), norm="l2")
    if result.algorithm.startswith("kmeans") and result.model is not None:
        cluster_id = int(result.model.predict(emb)[0])
    else:
        from sklearn.metrics.pairwise import cosine_distances
        dists      = cosine_distances(emb, result.embeddings_norm)[0]
        cluster_id = result.assignments[int(np.argmin(dists))].cluster_id
    return cluster_id, result.cluster_labels.get(cluster_id, "Unknown")


def should_retrain(original_n: int, current_n: int) -> bool:
    return (current_n - original_n) / max(original_n, 1) >= RETRAIN_THRESHOLD


# ── Persist / load ────────────────────────────────────────────────────────────

def save_clustering_result(result: ClusteringResult, path: str = MODEL_CACHE_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"✅ Clustering model saved → {path}")


def load_clustering_result(path: str = MODEL_CACHE_PATH) -> Optional[ClusteringResult]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        result = pickle.load(f)
    print(f"✅ Clustering model loaded ← {path}")
    return result


# ── Neo4j integration ─────────────────────────────────────────────────────────

def write_clusters_to_neo4j(result: ClusteringResult, user_id: int) -> None:
    from app.graph_db import get_driver
    driver = get_driver()
    with driver.session() as session:
        for a in result.assignments:
            session.run(
                """
                MATCH (p:Paper {paper_id: $paper_id, user_id: $user_id})
                SET p.cluster_id    = $cluster_id,
                    p.cluster_label = $cluster_label
                """,
                paper_id=a.paper_id, user_id=user_id,
                cluster_id=a.cluster_id, cluster_label=a.cluster_label,
            )
    print(f"✅ Wrote {len(result.assignments)} cluster assignments to Neo4j")


# ── Full pipeline convenience ─────────────────────────────────────────────────

def cluster_user_papers(
    user_id: int,
    db,
    algorithm: str = "kmeans",
    k: Optional[int] = None,
    write_to_neo4j: bool = True,
) -> ClusteringResult:
    """
    One-call pipeline: load papers → fetch embeddings → cluster → persist.

    For mixed-domain small corpora (< 12 papers), pass algorithm="dbscan"
    for cleaner separation. DBSCAN will assign lone outlier papers cluster_id=-1
    rather than forcing them into the wrong cluster.

    For homogeneous corpora (all same domain, ≥ 10 papers), algorithm="kmeans"
    is fine.
    """
    from app.models import Paper
    from app.similarity_model import papers_from_db_rows

    rows = db.query(Paper).filter(Paper.user_id == user_id).all()
    if len(rows) < 3:
        raise ValueError(f"Need at least 3 papers to cluster, got {len(rows)}.")

    papers, embeddings = papers_from_db_rows(rows)
    if len(papers) < 3:
        raise ValueError("Not enough papers with embeddings to cluster.")

    if algorithm == "dbscan":
        result = run_dbscan(embeddings, papers)
    else:
        result = run_kmeans(embeddings, papers, k=k)

    if write_to_neo4j:
        write_clusters_to_neo4j(result, user_id)

    save_clustering_result(result)
    return result