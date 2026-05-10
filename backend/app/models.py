from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime
from app.database import Base


# ─── SQLAlchemy ORM Models ────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    papers = relationship("Paper", back_populates="owner", cascade="all, delete-orphan")


class Paper(Base):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(String(36), unique=True, index=True, nullable=False)  # UUID
    title = Column(String(500), nullable=False)
    authors = Column(JSON, default=list)
    abstract = Column(Text, nullable=True)

    # Legacy field kept for backward compatibility
    summary = Column(Text, nullable=True)

    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    file_path = Column(String(500), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Structured content
    sections = Column(JSON, default=list)
    entities = Column(JSON, default=dict)
    topics = Column(JSON, default=list)
    section_summaries = Column(JSON, default=dict)
    tables_and_figures = Column(JSON, default=list)

    # FIX: processing status — "processing" | "ready" | "failed"
    # Allows frontend to poll /api/papers/{id}/status instead of guessing.
    status = Column(String(20), default="ready", nullable=False, server_default="ready")

    # Research-level structured analysis
    research_summary = Column(JSON, default=dict)
    research_gap = Column(JSON, default=dict)

    owner = relationship("User", back_populates="papers")


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    user_id: Optional[int] = None


class PaperUploadResponse(BaseModel):
    paper_id: str
    title: str
    message: str


class PaperListItem(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    topics: List[str]
    upload_date: datetime
    has_summary: bool
    # FIX: expose status so library page can show "Processing…" badge
    status: Optional[str] = "ready"

    class Config:
        from_attributes = True


class TableFigureItem(BaseModel):
    type: str
    label: str
    caption: str
    image_b64: Optional[str] = None


class SectionItem(BaseModel):
    title: str
    content: str
    level: Optional[int] = 1
    subsections: Optional[List["SectionItem"]] = []

    class Config:
        from_attributes = True

SectionItem.model_rebuild()


class ResearchSummary(BaseModel):
    problem_definition: Optional[str] = None
    background_and_prior_work: Optional[str] = None
    methodology: Optional[str] = None
    technical_novelty: Optional[str] = None
    experimental_setup: Optional[str] = None
    results_and_analysis: Optional[str] = None
    limitations: Optional[str] = None
    conclusion: Optional[str] = None


class ResearchGap(BaseModel):
    gap_identification: Optional[str] = None
    remaining_limitations: Optional[str] = None
    methodological_weaknesses: Optional[str] = None
    future_directions: Optional[List[str]] = []


class PaperDetails(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    abstract: Optional[str]
    summary: Optional[str] = None
    research_summary: Optional[Dict[str, Any]] = {}
    research_gap: Optional[Dict[str, Any]] = {}
    upload_date: datetime
    sections: List[SectionItem]
    entities: Dict[str, List[str]]
    topics: List[str]
    tables_and_figures: Optional[List[TableFigureItem]] = []
    # FIX: status + llm_status for frontend polling
    status: Optional[str] = "ready"
    llm_status: Optional[str] = "ok"

    class Config:
        from_attributes = True


class SearchResult(BaseModel):
    paper_id: str
    paper_title: str
    section: str
    text: str
    score: float


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    paper_id: Optional[str] = None
    chat_history: Optional[List[ChatMessage]] = []


class QueryResponse(BaseModel):
    answer: str
    sources: List[SearchResult]


class CompareRequest(BaseModel):
    paper_ids: List[str]


class ComparisonTable(BaseModel):
    problem: List[str] = []
    methodology: List[str] = []
    datasets: List[str] = []
    results: List[str] = []


class PaperStrengthWeakness(BaseModel):
    strengths: List[str] = []
    weaknesses: List[str] = []


class DeepCompareResponse(BaseModel):
    papers: List[Dict[str, Any]]
    comparison_table: ComparisonTable
    key_differences: List[str]
    problem_solution_relationship: str
    strengths_vs_weaknesses: Dict[str, PaperStrengthWeakness]
    best_use_cases: Dict[str, str]
    research_evolution: str
    similarity_scores: Optional[Dict[str, float]] = None
    shared_methods: List[str] = []
    shared_datasets: List[str] = []
    shared_topics: List[str] = []
    narrative: Optional[str] = None


class RecommendationItem(BaseModel):
    title: str
    authors: List[str]
    abstract: str
    source: str
    relevance: str
    url: Optional[str] = None