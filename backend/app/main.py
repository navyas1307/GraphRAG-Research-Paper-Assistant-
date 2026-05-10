
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, status, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import os
import shutil
import time
import threading
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from app.database import get_db, init_db, SessionLocal
from app.models import (
    User, Paper, UserCreate, UserResponse, Token,
    PaperUploadResponse, PaperListItem, PaperDetails, SectionItem,
    TableFigureItem, SearchResult, QueryRequest, QueryResponse,
    CompareRequest, DeepCompareResponse, RecommendationItem,
)
from app.auth import (
    verify_password, get_password_hash,
    create_access_token, get_current_user,
)
from app.pdf_processor import extract_text_from_pdf, create_chunks_from_sections
from app.embeddings import generate_embeddings, generate_single_embedding
from app.vector_db import init_collection, store_chunks, search_similar, delete_paper_chunks
from app.graph_db import (
    create_paper_graph, get_paper_graph, get_user_graph,
    find_related_papers, delete_paper_nodes,
    store_paper_embedding, create_similarity_edges,
)
from app import llm_service
from app.graphrag_compare import graphrag_compare

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Module-level caches ────────────────────────────────────────────────────────
# Keyed: {paper_id: {section_name: summary_text}}
_section_summary_cache: dict = {}

# Service availability flags (set on startup)
_qdrant_available = True
_neo4j_available  = True

app = FastAPI(
    title="GraphRAG Research Paper Assistant",
    description="AI-powered research paper management with semantic search, knowledge graphs, and recommendations",
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global _qdrant_available, _neo4j_available
    init_db()
    try:
        init_collection()
        _qdrant_available = True
    except Exception as e:
        _qdrant_available = False
        print(f"⚠️ Qdrant init failed — vector search disabled: {e}")
    try:
        from app.graph_db import get_driver
        get_driver().verify_connectivity()
        _neo4j_available = True
    except Exception as e:
        _neo4j_available = False
        print(f"⚠️ Neo4j init failed — graph features disabled: {e}")


# ─── Auth Endpoints ───────────────────────────────────────────────────────────

@app.post("/api/auth/signup", response_model=Token)
def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")

    user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=get_password_hash(user_data.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": str(user.id)})
    return Token(access_token=token, token_type="bearer")


@app.post("/api/auth/login", response_model=Token)
def login(user_data: UserCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_data.email).first()
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": str(user.id)})
    return Token(access_token=token, token_type="bearer")


@app.get("/api/auth/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user


# ─── Background Enrichment ────────────────────────────────────────────────────

def _background_enrich(paper_id: str, user_id: int, extracted: dict):
    """
    Runs embeddings + AI enrichment + graph + clustering in a background thread.
    Opens its own DB session (thread-safe). Updates status:
      "processing" → "ready" | "failed"

    Steps:
      1. Generate & store embeddings in Qdrant
      2. AI metadata extraction (entities, topics, research_summary)
      3. Build Neo4j entity graph
      4. Store paper-level embedding + create SIMILAR_TO edges (hybrid scored)
      5. Cluster / assign this paper to a research cluster
      6. SINGLE final db.commit() — writes ALL AI data + status="ready" atomically
         so the frontend never sees status="ready" with empty AI fields.
    """
    # BUG FIX: single session for the entire background job.
    # Previously there were two separate SessionLocal() calls — one for AI data
    # and one for status="ready" — which created a race on PostgreSQL READ COMMITTED
    # where the frontend could poll between the two commits and see ready+empty data.
    with SessionLocal() as db:
        try:
            # ── Step 1: Generate & store embeddings ───────────────────────────
            if _qdrant_available:
                try:
                    chunks = create_chunks_from_sections(extracted["sections"], paper_id)
                    if not chunks:
                        from app.pdf_processor import chunk_text
                        chunks = chunk_text(extracted["full_text"], paper_id, "body")
                    if chunks:
                        texts = [c["text"] for c in chunks]
                        embeddings = generate_embeddings(texts)
                        store_chunks(chunks, embeddings, user_id)
                except Exception as e:
                    print(f"⚠️ Embedding/Qdrant error (non-critical): {e}")

            print(f"🤖 Background: AI enrichment for {paper_id}...")

            # ── Step 2: AI metadata extraction ────────────────────────────────
            ai_metadata = llm_service.extract_all_metadata(
                title=extracted["title"],
                abstract=extracted.get("abstract") or "",
                full_text=extracted["full_text"],
                sections=extracted.get("sections", []),
            )

            # ── Step 3: Build Neo4j entity graph ──────────────────────────────
            if _neo4j_available:
                try:
                    create_paper_graph(
                        paper_id=paper_id,
                        title=extracted["title"],
                        authors=extracted.get("authors", []),
                        entities=ai_metadata.get("entities", {}),
                        topics=ai_metadata.get("topics", []),
                        user_id=user_id,
                    )
                except Exception as e:
                    print(f"⚠️ Neo4j graph error (non-critical): {e}")

            # ── Step 4: Store embedding + create hybrid SIMILAR_TO edges ──────
            paper_emb = None
            if _neo4j_available:
                try:
                    abstract_text = extracted.get("abstract") or ""
                    paper_text    = f"{extracted['title']}. {abstract_text}".strip()
                    paper_emb     = generate_single_embedding(paper_text)
                    store_paper_embedding(paper_id, paper_emb)
                    create_similarity_edges(
                        paper_id=paper_id,
                        user_id=user_id,
                        new_embedding=paper_emb,
                    )
                except Exception as e:
                    print(f"⚠️ Similarity edge error (non-critical): {e}")

            # ── Step 5: Cluster / assign paper ────────────────────────────────
            # Wrapped in its own try/except — clustering failure must NEVER
            # break the upload.
            if _neo4j_available:
                try:
                    from app.clustering_model import (
                        load_clustering_result,
                        assign_new_paper,
                        should_retrain,
                        cluster_user_papers,
                        save_clustering_result,
                    )

                    all_papers = db.query(Paper).filter(
                        Paper.user_id == user_id
                    ).all()
                    n_total = len(all_papers)

                    if n_total < 3:
                        print(f"ℹ️  Clustering skipped — only {n_total} paper(s) so far (need 3+)")
                    else:
                        cached_result = load_clustering_result()

                        if cached_result is None:
                            print(f"🔬 Clustering: first run (n={n_total})")
                            result = cluster_user_papers(
                                user_id=user_id,
                                db=db,
                                write_to_neo4j=True,
                            )
                            save_clustering_result(result)
                            print(f"✅ Clustering done: {result.n_clusters} clusters, "
                                  f"silhouette={result.silhouette_score:.3f}")

                        elif should_retrain(len(cached_result.assignments), n_total):
                            print(f"🔄 Clustering: retraining (was {len(cached_result.assignments)}, now {n_total})")
                            result = cluster_user_papers(
                                user_id=user_id,
                                db=db,
                                write_to_neo4j=True,
                            )
                            save_clustering_result(result)
                            print(f"✅ Clustering retrain done: {result.n_clusters} clusters")

                        else:
                            # Incremental: assign new paper to nearest existing cluster
                            if paper_emb is None:
                                abstract_text = extracted.get("abstract") or ""
                                paper_text    = f"{extracted['title']}. {abstract_text}".strip()
                                paper_emb     = generate_single_embedding(paper_text)
                            cid, clabel = assign_new_paper(paper_emb, cached_result)

                            from app.graph_db import get_driver
                            with get_driver().session() as sess:
                                sess.run(
                                    """
                                    MATCH (p:Paper {paper_id: $pid, user_id: $uid})
                                    SET p.cluster_id    = $cid,
                                        p.cluster_label = $clabel
                                    """,
                                    pid=paper_id,
                                    uid=user_id,
                                    cid=cid,
                                    clabel=clabel,
                                )
                            print(f"✅ Paper {paper_id[:8]} → cluster {cid}: '{clabel}' (incremental assign)")

                except Exception as e:
                    print(f"⚠️ Clustering step failed (non-critical): {e}")

            # ── Step 6: Write ALL AI data + status="ready" in ONE commit ──────
            # BUG FIX: previously AI data was committed in step 2 and
            # status="ready" was committed in a SEPARATE SessionLocal() at the
            # end. On PostgreSQL READ COMMITTED a frontend request landing between
            # those two commits sees ready+empty — this single commit eliminates
            # that entire race window.
            paper = db.query(Paper).filter(Paper.paper_id == paper_id).first()
            if paper:
                paper.research_summary  = ai_metadata.get("research_summary") or {}
                paper.research_gap      = ai_metadata.get("research_gap") or {}
                paper.entities          = ai_metadata.get("entities") or {}
                paper.topics            = ai_metadata.get("topics") or []
                paper.section_summaries = ai_metadata.get("section_summaries") or {}
                rs = paper.research_summary or {}
                paper.summary  = rs.get("problem_definition") or extracted.get("abstract") or ""
                paper.status   = "ready"
                db.commit()
                print(f"✅ Background enrichment complete for {paper_id}")

        except Exception as e:
            print(f"❌ Background enrichment failed for {paper_id}: {e}")
            try:
                paper = db.query(Paper).filter(Paper.paper_id == paper_id).first()
                if paper:
                    paper.status = "failed"
                    db.commit()
            except Exception:
                pass


# ─── Paper Upload / Re-enrich ─────────────────────────────────────────────────

@app.post("/api/papers/upload", response_model=PaperUploadResponse)
async def upload_paper(
    file: UploadFile = File(...),
    format_hint: str = Form(default="auto"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    paper_id  = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{paper_id}.pdf"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        print(f"📄 Processing PDF: {file.filename}")

        extracted = extract_text_from_pdf(str(file_path), fmt_hint=format_hint)

        if not extracted.get("full_text", "").strip():
            raise ValueError("Could not extract any text from PDF — file may be scanned/image-only.")

        paper = Paper(
            paper_id=paper_id,
            title=extracted["title"],
            authors=extracted.get("authors", []),
            abstract=extracted.get("abstract"),
            summary=None,
            research_summary={},
            research_gap={},
            file_path=str(file_path),
            user_id=current_user.id,
            sections=extracted.get("sections", []),
            entities=extracted.get("entities", {}),
            topics=[],
            tables_and_figures=extracted.get("tables_and_figures", []),
            status="processing",
        )
        db.add(paper)
        db.commit()

        threading.Thread(
            target=_background_enrich,
            args=(paper_id, current_user.id, extracted),
            daemon=True,
        ).start()

        return PaperUploadResponse(
            paper_id=paper_id,
            title=extracted["title"],
            message="Uploaded. AI summary, graph building, and clustering in background (~30s).",
        )

    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/papers/re-enrich-all")
def re_enrich_all_papers(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Re-enrich all papers. Uses FastAPI BackgroundTasks + 35s stagger between
    papers to stay well under the Gemini free-tier rate limit.
    """
    papers = db.query(Paper).filter(Paper.user_id == current_user.id).all()
    queued = []

    for idx, paper in enumerate(papers):
        if paper.file_path and Path(paper.file_path).exists():
            delay = idx * 35.0

            def _staggered(p=paper, d=delay):
                if d > 0:
                    time.sleep(d)
                try:
                    extracted = extract_text_from_pdf(p.file_path)
                    _background_enrich(p.paper_id, current_user.id, extracted)
                except Exception as e:
                    print(f"⚠️ Re-enrichment failed for {p.paper_id}: {e}")

            background_tasks.add_task(_staggered)
            queued.append(paper.paper_id)

    return {"message": f"Re-enrichment queued for {len(queued)} papers (staggered).", "queued": queued}


@app.post("/api/papers/{paper_id}/re-enrich")
def re_enrich_paper(
    paper_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    paper = db.query(Paper).filter(
        Paper.paper_id == paper_id,
        Paper.user_id == current_user.id
    ).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    if not paper.file_path or not Path(paper.file_path).exists():
        raise HTTPException(status_code=400, detail="Original PDF no longer available")

    try:
        extracted = extract_text_from_pdf(paper.file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF re-extraction failed: {e}")

    paper.status = "processing"
    db.commit()

    background_tasks.add_task(_background_enrich, paper_id, current_user.id, extracted)
    return {"message": "Re-enrichment started. Metadata will update in ~30s.", "paper_id": paper_id}


# ─── Paper List / Detail / Delete ─────────────────────────────────────────────

@app.get("/api/papers", response_model=List[PaperListItem])
def list_papers(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 20,
):
    papers = (
        db.query(Paper)
        .filter(Paper.user_id == current_user.id)
        .order_by(Paper.upload_date.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [
        PaperListItem(
            paper_id=p.paper_id,
            title=p.title,
            authors=p.authors or [],
            topics=p.topics or [],
            upload_date=p.upload_date,
            has_summary=bool(p.research_summary or p.summary),
            status=p.status or "ready",
        )
        for p in papers
    ]


@app.get("/api/papers/{paper_id}", response_model=PaperDetails)
def get_paper(
    paper_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns immediately from the DB — never blocks on AI enrichment.
    status + llm_status fields let the frontend poll for completion.
    """
    paper = db.query(Paper).filter(
        Paper.paper_id == paper_id,
        Paper.user_id == current_user.id
    ).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    def _build_section(s: dict) -> SectionItem:
        subs = [_build_section(sub) for sub in s.get("subsections", [])]
        return SectionItem(
            title=s.get("title", ""),
            content=s.get("content", ""),
            level=s.get("level", 1),
            subsections=subs,
        )

    def _build_figure(f: dict) -> TableFigureItem:
        return TableFigureItem(
            type=f.get("type", "figure"),
            label=f.get("label", ""),
            caption=f.get("caption", ""),
            image_b64=f.get("image_b64"),
        )

    return PaperDetails(
        paper_id=paper.paper_id,
        title=paper.title,
        authors=paper.authors or [],
        abstract=paper.abstract,
        summary=paper.summary,
        research_summary=paper.research_summary or {},
        research_gap=paper.research_gap or {},
        upload_date=paper.upload_date,
        sections=[_build_section(s) for s in (paper.sections or [])],
        entities=paper.entities or {},
        topics=paper.topics or [],
        tables_and_figures=[_build_figure(f) for f in (paper.tables_and_figures or [])],
        status=paper.status or "ready",
        llm_status="ok",
    )


@app.get("/api/papers/{paper_id}/status")
def get_paper_status(
    paper_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Lightweight polling endpoint — returns status + has_summary + has_ai_data.

    has_ai_data is TRUE only when actual AI content fields are populated.
    It uses real content checks (not bool({})) so the frontend can reliably
    detect the "stale-ready" state (status=ready but AI data not yet visible)
    and keep polling until content arrives.

    Frontend should poll every 5s while:
      status === "processing"  OR  (status === "ready" AND !has_ai_data)
    """
    paper = db.query(Paper).filter(
        Paper.paper_id == paper_id,
        Paper.user_id == current_user.id
    ).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    # BUG FIX: check actual content, not bool({}) which is always False for
    # empty dicts and was masking successful AI runs.
    rs = paper.research_summary or {}
    ents = paper.entities or {}
    has_ai_data = bool(
        rs.get("methodology") or
        rs.get("problem_definition") or
        rs.get("results_and_analysis") or
        ents.get("methods") or
        ents.get("topics") or
        (paper.topics or [])
    )

    return {
        "paper_id":    paper_id,
        "status":      paper.status or "ready",
        "has_summary": bool(paper.research_summary or paper.summary),
        "has_ai_data": has_ai_data,
        "llm_status":  "ok",
    }


@app.delete("/api/papers/{paper_id}")
def delete_paper(
    paper_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    paper = db.query(Paper).filter(
        Paper.paper_id == paper_id,
        Paper.user_id == current_user.id
    ).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    try:
        delete_paper_chunks(paper_id)
    except Exception:
        pass
    try:
        delete_paper_nodes(paper_id)
    except Exception:
        pass
    if Path(paper.file_path).exists():
        Path(paper.file_path).unlink()

    db.delete(paper)
    db.commit()
    return {"message": "Paper deleted successfully"}


# ─── Section Summary ──────────────────────────────────────────────────────────

@app.get("/api/papers/{paper_id}/section-summary")
def get_section_summary(
    paper_id: str,
    section_name: str = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Three-layer cache — zero on-demand LLM calls.
    1. In-memory process cache  (instant)
    2. Pre-built from background enrichment in DB  (instant)
    3. Extractive fallback  (no LLM, instant)
    """
    paper = db.query(Paper).filter(
        Paper.paper_id == paper_id,
        Paper.user_id == current_user.id
    ).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    # 1. In-memory cache
    cached = _section_summary_cache.get(paper_id, {}).get(section_name)
    if cached:
        return {"summary": cached}

    # 2. Pre-built summaries from background enrichment
    section_summaries = paper.section_summaries or {}
    if section_name in section_summaries:
        _section_summary_cache.setdefault(paper_id, {})[section_name] = section_summaries[section_name]
        return {"summary": section_summaries[section_name]}

    # 3. Extractive fallback — no LLM call
    section = next((s for s in (paper.sections or []) if s["title"] == section_name), None)
    if not section:
        raise HTTPException(status_code=404, detail="Section not found")

    content = section.get("content", "")
    sentences = content.replace("\n", " ").split(". ")
    summary = ". ".join(sentences[:4]).strip()
    if summary and not summary.endswith("."):
        summary += "."

    _section_summary_cache.setdefault(paper_id, {})[section_name] = summary
    return {"summary": summary}


# ─── Graph ────────────────────────────────────────────────────────────────────

@app.get("/api/papers/{paper_id}/graph")
def get_paper_graph_endpoint(
    paper_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    paper = db.query(Paper).filter(
        Paper.paper_id == paper_id,
        Paper.user_id == current_user.id
    ).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    if not _neo4j_available:
        return {"nodes": [], "edges": [], "message": "Graph service unavailable"}

    try:
        graph_data = get_paper_graph(paper_id)
        return graph_data
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}


@app.get("/api/graph")
def get_full_graph(
    current_user: User = Depends(get_current_user),
):
    if not _neo4j_available:
        return {"nodes": [], "edges": [], "message": "Graph service unavailable"}

    try:
        graph_data = get_user_graph(current_user.id)
        return graph_data
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}


# ─── Search & Query ───────────────────────────────────────────────────────────

@app.get("/api/search", response_model=List[SearchResult])
def search_papers(
    query: str,
    paper_id: Optional[str] = None,
    top_k: int = 5,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not _qdrant_available:
        raise HTTPException(status_code=503, detail="Vector search service unavailable")

    query_embedding = generate_single_embedding(query)
    results = search_similar(
        query_embedding=query_embedding,
        user_id=current_user.id,
        paper_id=paper_id,
        top_k=top_k,
    )

    enriched = []
    for r in results:
        paper = db.query(Paper).filter(Paper.paper_id == r["paper_id"]).first()
        enriched.append(SearchResult(
            paper_id=r["paper_id"],
            paper_title=paper.title if paper else "Unknown",
            section=r["section"],
            text=r["text"],
            score=r["score"],
        ))
    return enriched


@app.post("/api/query", response_model=QueryResponse)
def query_papers(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not _qdrant_available:
        raise HTTPException(status_code=503, detail="Vector search service unavailable")

    query_embedding = generate_single_embedding(request.question)
    search_results  = search_similar(
        query_embedding=query_embedding,
        user_id=current_user.id,
        paper_id=request.paper_id,
        top_k=5,
    )

    context_papers = []
    if request.paper_id and _neo4j_available:
        try:
            related = find_related_papers(request.paper_id, user_id=current_user.id, limit=3)  # FIX: was missing user_id
            paper_ids = [r["paper_id"] for r in related]
            for pid in paper_ids:
                p = db.query(Paper).filter(Paper.paper_id == pid).first()
                if p:
                    context_papers.append({
                        "paper_id": p.paper_id,
                        "title": p.title,
                        "entities": p.entities or {},
                        "topics": p.topics or [],
                        "research_summary": p.research_summary or {},
                    })
        except Exception:
            pass

    answer = llm_service.answer_question(
        question=request.question,
        search_results=search_results,
        chat_history=[m.dict() for m in (request.chat_history or [])],
        context_papers=context_papers,
    )

    enriched_sources = []
    for r in search_results:
        paper = db.query(Paper).filter(Paper.paper_id == r["paper_id"]).first()
        enriched_sources.append(SearchResult(
            paper_id=r["paper_id"],
            paper_title=paper.title if paper else "Unknown",
            section=r["section"],
            text=r["text"],
            score=r["score"],
        ))

    return QueryResponse(answer=answer, sources=enriched_sources)


# ─── Compare ─────────────────────────────────────────────────────────────────

@app.post("/api/compare")
def compare_papers(
    request: CompareRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if len(request.paper_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 papers to compare")

    papers = []
    for pid in request.paper_ids:
        p = db.query(Paper).filter(
            Paper.paper_id == pid,
            Paper.user_id == current_user.id,
        ).first()
        if not p:
            raise HTTPException(status_code=404, detail=f"Paper {pid} not found")
        papers.append(p)

    comparison = llm_service.compare_papers([{
        "paper_id": p.paper_id,
        "title": p.title,
        "research_summary": p.research_summary or {},
        "entities": p.entities or {},
        "topics": p.topics or [],
    } for p in papers])

    return comparison


@app.post("/api/papers/compare", response_model=DeepCompareResponse)
def deep_compare_papers(
    request: CompareRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if len(request.paper_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 papers to compare")

    # FIX: graphrag_compare() expects (paper_ids, orm_papers, user_id, db).
    # The old code built plain dicts and called graphrag_compare(papers_data)
    # with only 1 arg — TypeError: missing 3 required positional arguments.
    # Now we pass ORM objects with the correct 4-argument signature.
    orm_papers = []
    for pid in request.paper_ids:
        p = db.query(Paper).filter(
            Paper.paper_id == pid,
            Paper.user_id == current_user.id,
        ).first()
        if not p:
            raise HTTPException(status_code=404, detail=f"Paper {pid} not found")
        orm_papers.append(p)

    return graphrag_compare(
        paper_ids=request.paper_ids,
        papers=orm_papers,
        user_id=current_user.id,
        db=db,
    )


# ─── Recommendations ──────────────────────────────────────────────────────────

@app.get("/api/recommendations/{paper_id}")
def get_recommendations(
    paper_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    paper = db.query(Paper).filter(
        Paper.paper_id == paper_id,
        Paper.user_id == current_user.id,
    ).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    local_recs = []
    if _neo4j_available:
        try:
            related = find_related_papers(paper_id, user_id=current_user.id, limit=5)  # FIX: was missing user_id
            for r in related:
                p = db.query(Paper).filter(Paper.paper_id == r["paper_id"]).first()
                if p:
                    # Build a descriptive relevance string using shared entities
                    sim  = r.get("similarity_score", r.get("score", 0))
                    shared = r.get("shared_items") or []
                    if shared:
                        rel_str = f"Shared: {', '.join(str(s) for s in shared[:3])} · score {sim:.2f}"
                    else:
                        rel_str = f"Semantic similarity: {sim:.2f}"
                    local_recs.append(RecommendationItem(
                        title=p.title,
                        authors=p.authors or [],
                        abstract=(p.abstract or "")[:300] + "…",
                        source="library",
                        relevance=rel_str,
                    ))
        except Exception:
            pass

    arxiv_recs = []
    try:
        import arxiv
        import concurrent.futures

        # FIX: build a precise arXiv query from title + topics rather than
        # concatenating raw topic strings. The old approach joined topics like
        # "chocolate health effects cocoa health effects polyphenols" into one
        # blob, which returned unrelated health papers (AI/digital health, EHR,
        # etc.) instead of actual chocolate/cocoa research.
        #
        # New strategy:
        #   1. Use the full paper title as the primary anchor (most precise).
        #   2. Append up to 2 short topics as additional keywords.
        #   3. Strip pure hex-looking tokens (paper IDs) as before.
        # This keeps the query focused on the actual paper's subject.
        _stop = {"based", "using", "towards", "approach", "study", "analysis",
                 "paper", "novel", "large", "model", "learning", "deep", "review",
                 "research", "survey", "system", "their", "with", "from", "that",
                 "this", "these", "which", "about", "through", "international",
                 "journal", "conference", "proceedings"}

        def _clean_token(w: str) -> bool:
            if len(w) <= 3:
                return False
            if all(c in "0123456789abcdefABCDEF-" for c in w):
                return False
            return w.lower() not in _stop

        # Title words as primary query
        title_tokens = [w for w in paper.title.split() if _clean_token(w)]
        title_query = " ".join(title_tokens[:10])

        # Add up to 2 high-signal topics that aren't already in the title
        title_lower = paper.title.lower()
        extra_topics = [
            t for t in (paper.topics or [])[:5]
            if t.lower() not in title_lower and _clean_token(t)
        ][:2]

        search_terms = title_query
        if extra_topics:
            search_terms = f"{title_query} {' '.join(extra_topics)}"

        if not search_terms.strip():
            search_terms = paper.title[:80]

        def _fetch_arxiv():
            try:
                client = arxiv.Client(page_size=5, delay_seconds=1, num_retries=1)
                search = arxiv.Search(
                    query=search_terms, max_results=5,
                    sort_by=arxiv.SortCriterion.Relevance,
                )
                return list(client.results(search))
            except AttributeError:
                search = arxiv.Search(
                    query=search_terms, max_results=5,
                    sort_by=arxiv.SortCriterion.Relevance,
                )
                return list(search.results())

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_fetch_arxiv)
            try:
                results = future.result(timeout=8)
                for result in results:
                    arxiv_recs.append(RecommendationItem(
                        title=result.title,
                        authors=[str(a) for a in result.authors[:3]],
                        abstract=(result.summary or "")[:300] + "…",
                        source="arxiv",
                        relevance=f"arXiv · related to: {search_terms[:60]}",
                        url=result.entry_id,
                    ))
            except concurrent.futures.TimeoutError:
                print("arXiv search timed out after 8s — skipping")
    except Exception as e:
        print(f"arXiv search error: {e}")

    return {"local": local_recs, "arxiv": arxiv_recs}


# ─── Figure Summary ──────────────────────────────────────────────────────────

@app.get("/api/papers/{paper_id}/figure-summary")
def get_figure_summary(
    paper_id: str,
    label: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    paper = db.query(Paper).filter(
        Paper.paper_id == paper_id,
        Paper.user_id == current_user.id
    ).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    items = paper.tables_and_figures or []
    item  = next((i for i in items if i.get("label") == label), None)
    if not item:
        raise HTTPException(status_code=404, detail="Figure/table not found")

    cache_key = f"fig::{label}"
    cached = _section_summary_cache.get(paper_id, {}).get(cache_key)
    if cached:
        return {"summary": cached}

    summary = llm_service.generate_figure_summary(
        label=item["label"],
        caption=item["caption"],
        item_type=item["type"],
        paper_title=paper.title,
    )

    _section_summary_cache.setdefault(paper_id, {})[cache_key] = summary
    return {"summary": summary}


# ─── Rebuild similarity edges ─────────────────────────────────────────────────

@app.post("/api/papers/rebuild-similarity")
def rebuild_similarity_edges(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    papers = db.query(Paper).filter(Paper.user_id == current_user.id).all()

    def _rebuild():
        print(f"🔧 Rebuilding hybrid similarity edges for {len(papers)} papers...")
        with SessionLocal() as db2:
            for i, p in enumerate(papers):
                try:
                    text = f"{p.title}. {p.abstract or ''}".strip()
                    emb  = generate_single_embedding(text)
                    store_paper_embedding(p.paper_id, emb)
                    created = create_similarity_edges(
                        paper_id=p.paper_id,
                        user_id=current_user.id,
                        new_embedding=emb,
                    )
                    print(f"  [{i+1}/{len(papers)}] {p.paper_id[:8]}… → {len(created)} edges")
                except Exception as e:
                    print(f"  ⚠️  Failed for {p.paper_id[:8]}: {e}")
        print("✅ Hybrid similarity edge rebuild complete")

    background_tasks.add_task(_rebuild)
    return {
        "message": f"Rebuilding hybrid similarity edges for {len(papers)} papers in background.",
        "paper_ids": [p.paper_id for p in papers],
    }


# ─── Cluster AI Summary ───────────────────────────────────────────────────────

@app.post("/api/cluster-summary")
def get_cluster_summary(
    body: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Generate an AI summary explaining WHY a set of papers are in the same cluster.

    Request body:
        paper_ids     : list[str]  — paper_ids of all papers in this cluster
        cluster_label : str        — the auto-generated cluster label
        signal_type   : str        — "strong" | "weak" | "none"

    Returns:
        { summary: str, n_papers: int }

    This endpoint exists because:
    1. /api/compare is designed for user-selected pairs (structured JSON output).
    2. compare_papers_narrative() has no cluster signal injection, causing the
       LLM to say "cannot identify a theme" for semantically-clustered papers.
    3. Papers with missing research_summary would disappear from the prompt —
       summarize_cluster() always falls back to abstract/title to include them.
    """
    paper_ids     = body.get("paper_ids") or []
    cluster_label = body.get("cluster_label", "Research Cluster")
    signal_type   = body.get("signal_type", "none")

    if not paper_ids:
        raise HTTPException(status_code=400, detail="paper_ids is required")

    papers = db.query(Paper).filter(
        Paper.paper_id.in_(paper_ids),
        Paper.user_id == current_user.id,
    ).all()

    if not papers:
        raise HTTPException(status_code=404, detail="No papers found for given IDs")

    papers_data = [
        {
            "title":            p.title,
            "abstract":         p.abstract or "",
            "entities":         p.entities or {},
            "topics":           p.topics or [],
            "research_summary": p.research_summary or {},
        }
        for p in papers
    ]

    summary = llm_service.summarize_cluster(papers_data, cluster_label, signal_type)
    return {"summary": summary, "n_papers": len(papers)}


# ─── Force re-cluster ─────────────────────────────────────────────────────────

@app.post("/api/papers/recluster")
def recluster_papers(
    background_tasks: BackgroundTasks,
    algorithm: str = Query(default="kmeans", description="kmeans or dbscan"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    papers = db.query(Paper).filter(Paper.user_id == current_user.id).all()
    if len(papers) < 3:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 3 papers to cluster, found {len(papers)}."
        )

    def _do_recluster():
        from app.clustering_model import cluster_user_papers, save_clustering_result
        with SessionLocal() as db2:
            try:
                result = cluster_user_papers(
                    user_id=current_user.id,
                    db=db2,
                    algorithm=algorithm,
                    write_to_neo4j=True,
                )
                save_clustering_result(result)
                print(
                    f"✅ Recluster complete: {result.n_clusters} clusters, "
                    f"silhouette={result.silhouette_score:.3f} ({algorithm})"
                )
            except Exception as e:
                print(f"❌ Recluster failed: {e}")

    background_tasks.add_task(_do_recluster)
    return {
        "message": f"Re-clustering started ({algorithm}) for {len(papers)} papers in background.",
        "paper_count": len(papers),
        "algorithm": algorithm,
    }


# ─── Automatic Evaluation ────────────────────────────────────────────────────

from app.evaluation import run_full_evaluation

@app.get("/api/evaluate")
def evaluate(db: Session = Depends(get_db)):
    user_id = 1  # hardcode temporarily
    return run_full_evaluation(user_id, db)

# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":    "ok",
        "version":   "2.2.0",
        "qdrant":    "up" if _qdrant_available else "down",
        "neo4j":     "up" if _neo4j_available  else "down",
        "llm_quota": "ok",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)