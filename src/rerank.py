"""
rerank.py
Stage 2a: Score each (JD, candidate_profile) pair with a cross-encoder.
Model: cross-encoder/ms-marco-MiniLM-L-6-v2
CPU-compatible, 68MB, scores 500 pairs in ~60-90 seconds.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.jd_parser import JDSpec
from src.utils import truncate_to_tokens


CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_BATCH_SIZE = 32


def build_candidate_text(row: pd.Series) -> str:
    """
    Construct the candidate text fed to the cross-encoder.
    Keep total length under 400 tokens.
    """
    parts = []

    title = row.get("current_title", "")
    company = row.get("current_company", "")
    if title:
        parts.append(f"Title: {title}")
    if company:
        parts.append(f"at {company}")

    yoe = row.get("years_of_experience", 0)
    parts.append(f"Experience: {yoe} years")

    # Skills (cap at 20)
    try:
        skills = json.loads(row.get("skills", "[]"))
    except (json.JSONDecodeError, TypeError):
        skills = []
    if skills:
        parts.append(f"Skills: {', '.join(skills[:20])}")

    edu_degree = row.get("education_degree", "")
    edu_field = row.get("education_field", "")
    if edu_degree or edu_field:
        parts.append(f"Education: {edu_degree} {edu_field}".strip())

    location = row.get("location_city", "")
    country = row.get("location_country", "")
    if location or country:
        parts.append(f"Location: {location}, {country}".strip(", "))

    industry = row.get("current_industry", "")
    if industry:
        parts.append(f"Industry: {industry}")

    summary = row.get("summary", "")
    if summary:
        parts.append(summary[:300])

    text = " | ".join(parts)
    return truncate_to_tokens(text, max_tokens=400)


def cross_encode(
    jd_spec: JDSpec,
    pool_df: pd.DataFrame,
    model: CrossEncoder | None = None,
) -> np.ndarray:
    """
    For each row in pool_df, build (jd_text, candidate_text) pair.
    Batch-predict with model.predict().
    Returns numpy array of raw CE scores.
    """
    t0 = time.time()

    if model is None:
        print(f"[rerank] Loading cross-encoder {CE_MODEL}...")
        model = CrossEncoder(CE_MODEL, max_length=512, device="cpu")

    # Truncate JD to ~500 chars for cross-encoder input
    jd_text = truncate_to_tokens(jd_spec.raw_text, max_tokens=250)

    # Build pairs
    pairs = []
    for _, row in pool_df.iterrows():
        candidate_text = build_candidate_text(row)
        pairs.append((jd_text, candidate_text))

    print(f"[rerank] Cross-encoding {len(pairs)} pairs with batch_size={CE_BATCH_SIZE}...")

    # Batch predict
    scores = model.predict(
        pairs,
        batch_size=CE_BATCH_SIZE,
        show_progress_bar=True,
    )

    scores = np.array(scores, dtype=np.float32)
    elapsed = time.time() - t0
    print(f"[rerank] Cross-encoded {len(pairs)} pairs in {elapsed:.1f}s")
    print(f"[rerank] Score range: [{scores.min():.4f}, {scores.max():.4f}], "
          f"mean={scores.mean():.4f}")
    return scores
