"""
retrieve.py
Stage 1: Given a JDSpec, retrieve top-500 candidate indices via
FAISS + BM25 dual retrieval with Reciprocal Rank Fusion (RRF).
"""

import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.jd_parser import JDSpec
from src.utils import tokenize_text, education_meets_requirement


FAISS_TOP_K = 2000      # retrieve from FAISS
BM25_TOP_K = 2000       # retrieve from BM25
RRF_K = 60              # RRF smoothing constant
FINAL_POOL_SIZE = 700   # candidates passed to cross-encoder


def faiss_retrieve(
    jd_spec: JDSpec,
    faiss_index: faiss.Index,
    model: SentenceTransformer,
) -> list[tuple[int, float]]:
    """
    Embed jd_spec.query_text with the same bi-encoder model.
    L2-normalize the query vector.
    Search faiss_index for top FAISS_TOP_K.
    Returns list of (candidate_array_index, similarity_score).
    """
    t0 = time.time()

    # Encode and normalize
    query_vec = model.encode(
        [jd_spec.query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Search
    faiss.omp_set_num_threads(1)  # Determinism
    D, I = faiss_index.search(query_vec, FAISS_TOP_K)

    results = []
    for idx, score in zip(I[0], D[0]):
        if idx >= 0:  # FAISS returns -1 for missing
            results.append((int(idx), float(score)))

    elapsed = time.time() - t0
    print(f"[retrieve] FAISS retrieved {len(results)} candidates in {elapsed:.2f}s")
    return results


def bm25_retrieve(
    jd_spec: JDSpec,
    bm25,
    tokenize_fn=None,
) -> list[tuple[int, float]]:
    """
    Tokenize jd_spec.query_text, get BM25 scores for all documents.
    Returns top BM25_TOP_K as list of (array_index, bm25_score).
    """
    t0 = time.time()

    if tokenize_fn is None:
        tokenize_fn = tokenize_text

    query_tokens = tokenize_fn(jd_spec.query_text)

    # Get scores for all documents
    scores = bm25.get_scores(query_tokens)

    # Get top-K indices
    top_indices = np.argsort(scores)[::-1][:BM25_TOP_K]
    results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    elapsed = time.time() - t0
    print(f"[retrieve] BM25 retrieved {len(results)} candidates in {elapsed:.2f}s")
    return results


def reciprocal_rank_fusion(
    faiss_results: list[tuple[int, float]],
    bm25_results: list[tuple[int, float]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion: for each result list, compute RRF score = 1/(k + rank).
    Merge by summing RRF scores from both lists for each unique candidate.
    Returns top FINAL_POOL_SIZE sorted by RRF score descending.
    """
    t0 = time.time()
    rrf_scores: dict[int, float] = {}

    # FAISS RRF scores
    for rank, (idx, _score) in enumerate(faiss_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)

    # BM25 RRF scores
    for rank, (idx, _score) in enumerate(bm25_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)

    # Sort by RRF score descending
    sorted_results = sorted(rrf_scores.items(), key=lambda x: -x[1])
    results = sorted_results[:FINAL_POOL_SIZE]

    elapsed = time.time() - t0
    n_both = sum(1 for idx in rrf_scores
                 if any(r[0] == idx for r in faiss_results)
                 and any(r[0] == idx for r in bm25_results))
    print(f"[retrieve] RRF fusion: {len(rrf_scores)} unique candidates, "
          f"{n_both} in both lists, returning top {len(results)} in {elapsed:.2f}s")
    return results


def apply_hard_knockouts(
    pool: list[tuple[int, float]],
    df: pd.DataFrame,
    jd_spec: JDSpec,
) -> list[tuple[int, float]]:
    """
    Remove candidates from pool who fail hard knockout rules.
    Only hard impossibilities — if applying knockouts would reduce pool
    below 150, relax the constraint.
    """
    t0 = time.time()
    original_size = len(pool)
    removed = 0

    filtered = []
    for idx, score in pool:
        row = df.iloc[idx]
        keep = True

        # YOE knockout
        if jd_spec.min_years_experience is not None:
            yoe = row.get("years_of_experience", 0)
            if yoe < jd_spec.min_years_experience:
                keep = False

        # Education knockout
        if keep and jd_spec.required_education:
            edu = row.get("education_degree", "")
            if not education_meets_requirement(edu, jd_spec.required_education):
                keep = False

        # Required skills: at least 1 overlap (lenient)
        if keep and jd_spec.required_skills:
            try:
                candidate_skills = json.loads(row.get("skills", "[]"))
            except (json.JSONDecodeError, TypeError):
                candidate_skills = []
            candidate_skills_lower = {s.lower() for s in candidate_skills}
            has_any = any(
                req.lower() in " ".join(candidate_skills_lower)
                for req in jd_spec.required_skills
            )
            # Only knockout if zero overlap AND title doesn't match
            title_lower = row.get("current_title", "").lower()
            title_relevant = any(
                kw in title_lower
                for kw in ["engineer", "ml", "ai", "data", "scientist",
                           "developer", "architect", "research"]
            )
            if not has_any and not title_relevant:
                keep = False

        if keep:
            filtered.append((idx, score))
        else:
            removed += 1

    # Safety: if too aggressive, relax
    if len(filtered) < 150 and original_size >= 150:
        print(f"[retrieve] WARNING: knockouts too aggressive ({len(filtered)} remaining), "
              f"relaxing to keep top {min(300, original_size)}")
        filtered = pool[:min(300, original_size)]
        removed = original_size - len(filtered)

    elapsed = time.time() - t0
    print(f"[retrieve] Hard knockouts: removed {removed}/{original_size}, "
          f"{len(filtered)} remaining in {elapsed:.2f}s")
    return filtered
