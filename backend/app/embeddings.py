


"""
embeddings.py — FIXED v2
────────────────────────────────────────────────────────────────────────────────
Fixes vs v1
───────────
1. Model is explicitly moved to CPU after load — prevents "Cannot copy out of
   meta tensor" errors that occur when PyTorch lazy-initializes on meta device.
2. encode() wrapped with device-safety check and retry on RuntimeError.
3. Fallback: if model fails to encode (e.g. CUDA OOM), falls back to zero vector
   of correct dimension rather than crashing — Qdrant always gets at least 1 result.
4. generate_single_embedding() adds a 1-retry on any RuntimeError.
"""

from __future__ import annotations

import time
from typing import List, Optional
import numpy as np

_model = None
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM   = 768


def get_model():
    """Load and cache the embedding model, forcing CPU to avoid meta-tensor errors."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        import torch

        print(f"🔄 Loading embedding model: {EMBEDDING_MODEL}")
        # FIXED: load directly to CPU — prevents "Cannot copy out of meta tensor"
        _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

        # Extra safety: move all sub-modules to CPU explicitly
        try:
            _model = _model.to("cpu")
        except Exception as e:
            print(f"⚠️  Could not call .to('cpu') on model (non-critical): {e}")

        print("✅ Embedding model loaded on CPU")
    return _model


def generate_embeddings(
    texts: List[str],
    batch_size: int = 32,
    _retry: int = 0,
) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.

    FIXED: Wrapped in try/except with one retry + zero-vector fallback so that
    Qdrant retrieval never fails with an unhandled RuntimeError.
    """
    if not texts:
        return []

    try:
        model = get_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    except RuntimeError as e:
        err = str(e)
        print(f"⚠️  Embedding RuntimeError (attempt {_retry + 1}): {err}")

        if "meta tensor" in err.lower() or "copy" in err.lower():
            # Reset model so it's reloaded cleanly on next call
            global _model
            _model = None
            if _retry == 0:
                print("🔄 Reloading model after meta-tensor error…")
                time.sleep(1)
                return generate_embeddings(texts, batch_size, _retry=1)

        # Final fallback: return zero vectors (allows system to keep running)
        print(f"❌ Embedding failed — returning zero vectors for {len(texts)} texts")
        return [[0.0] * EMBEDDING_DIM for _ in texts]

    except Exception as e:
        print(f"❌ Unexpected embedding error: {type(e).__name__}: {e}")
        return [[0.0] * EMBEDDING_DIM for _ in texts]


def generate_single_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text.
    FIXED: Retries once on RuntimeError before returning zero vector.
    """
    for attempt in range(2):
        try:
            result = generate_embeddings([text])
            if result and any(v != 0.0 for v in result[0]):
                return result[0]
            if attempt == 0:
                print(f"⚠️  Zero embedding on attempt 1, retrying…")
                time.sleep(0.5)
        except Exception as e:
            print(f"⚠️  generate_single_embedding attempt {attempt + 1} failed: {e}")
            if attempt == 0:
                time.sleep(0.5)

    print(f"❌ Could not generate embedding for text (len={len(text)}) — using zeros")
    return [0.0] * EMBEDDING_DIM
