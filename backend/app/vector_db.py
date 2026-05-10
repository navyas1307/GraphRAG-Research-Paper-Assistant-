from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, SearchRequest
)
import uuid
import os
from dotenv import load_dotenv

from app.embeddings import EMBEDDING_DIM

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "research_papers"

_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


def init_collection():
    """Create the Qdrant collection if it doesn't exist."""
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"✅ Created Qdrant collection: {COLLECTION_NAME}")
    else:
        print(f"✅ Qdrant collection already exists: {COLLECTION_NAME}")


def store_chunks(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    user_id: int,
):
    """Store text chunks with their embeddings in Qdrant."""
    client = get_client()
    points = []

    for chunk, embedding in zip(chunks, embeddings):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "paper_id": chunk["paper_id"],
                "user_id": user_id,
                "section": chunk["section"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            }
        )
        points.append(point)

    # Batch upsert
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i:i + batch_size],
        )

    print(f"✅ Stored {len(points)} chunks in Qdrant")


def search_similar(
    query_embedding: List[float],
    user_id: int,
    paper_id: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Search for similar chunks."""
    client = get_client()

    # Build filter
    must_conditions = [
        FieldCondition(key="user_id", match=MatchValue(value=user_id))
    ]
    if paper_id:
        must_conditions.append(
            FieldCondition(key="paper_id", match=MatchValue(value=paper_id))
        )

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        query_filter=Filter(must=must_conditions),
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "paper_id": r.payload["paper_id"],
            "section": r.payload["section"],
            "text": r.payload["text"],
            "score": r.score,
        }
        for r in results
    ]


def delete_paper_chunks(paper_id: str):
    """Delete all chunks for a paper."""
    client = get_client()
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
        ),
    )
    print(f"✅ Deleted chunks for paper: {paper_id}")
