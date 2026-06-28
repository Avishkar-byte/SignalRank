"""
score_fusion.py
Stage 2b: Combine CE score, behavioral signal score, BM25 score, and honeypot penalty.

Final score formula:
  composite = (
      0.55 * ce_score_norm
    + 0.30 * signal_score
    + 0.15 * bm25_score_norm
    - 0.90 * honeypot_flag
  )
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.jd_parser import JDSpec


# Weight constants
W_CE = 0.55
W_SIGNAL = 0.30
W_BM25 = 0.15
HONEYPOT_PENALTY = 0.90


def min_max_norm(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1]. Handle edge case where min == max."""
    vmin = arr.min()
    vmax = arr.max()
    if vmax - vmin < 1e-10:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def compute_skill_match_bonus(
    pool_df: pd.DataFrame,
    jd_spec: JDSpec,
) -> np.ndarray:
    """
    Compute a skill-match bonus score for each candidate in the pool.
    This gives extra weight to candidates whose skills directly match JD requirements.
    Returns array of scores in [0, 1].
    """
    scores = np.zeros(len(pool_df), dtype=np.float32)

    for i, (_, row) in enumerate(pool_df.iterrows()):
        try:
            skills = json.loads(row.get("skills", "[]"))
        except (json.JSONDecodeError, TypeError):
            skills = []
        skills_lower = {s.lower() for s in skills}

        # Check matches against skill_weight_map
        total_weight = 0.0
        matched_weight = 0.0
        for skill, weight in jd_spec.skill_weight_map.items():
            total_weight += weight
            # Fuzzy match: check if any candidate skill contains the JD skill keyword
            if any(skill in cs for cs in skills_lower):
                matched_weight += weight

        if total_weight > 0:
            scores[i] = matched_weight / total_weight

        # Title relevance bonus
        title = row.get("current_title", "").lower()
        title_keywords = ["engineer", "ml", "ai", "machine learning", "data scientist",
                         "research", "nlp", "developer", "architect"]
        if any(kw in title for kw in title_keywords):
            scores[i] = min(1.0, scores[i] + 0.15)

        # Consulting company penalty
        company = row.get("current_company", "").lower()
        if jd_spec.disqualifier_companies:
            if any(dc in company for dc in jd_spec.disqualifier_companies):
                scores[i] = max(0.0, scores[i] - 0.20)

    return scores


def compute_recency_multiplier(candidate_signals: np.ndarray, signal_names: list[str]) -> float:
    """
    Identify the recency-related signals.
    Days since active (signal 19) is inverted, so higher value = more recent.
    """
    try:
        idx = signal_names.index("days_since_active")
        val = candidate_signals[idx]
        if val > 0.7:
            return 1.05
        elif val >= 0.4:
            return 1.0
        else:
            return 0.97
    except ValueError:
        return 1.0


def compute_composite_scores(
    pool_indices: list[int],
    pool_df: pd.DataFrame,
    ce_scores: np.ndarray,
    signal_matrix: np.ndarray,
    bm25_rrf_scores: np.ndarray,
    honeypot_penalties: np.ndarray,
    jd_spec: JDSpec,
    signal_names: list[str] = None,
) -> pd.DataFrame:
    """
    Assemble composite scores from all components.
    """
    if signal_names is None:
        from src.signal_extractor import SIGNAL_NAMES
        signal_names = SIGNAL_NAMES

    # Normalize CE scores across pool
    ce_norm = min_max_norm(ce_scores)

    # Signal score: dot product of candidate signals with JD signal weights
    signal_scores = np.zeros(len(pool_indices), dtype=np.float32)
    recency_mults = np.ones(len(pool_indices), dtype=np.float32)

    for i, idx in enumerate(pool_indices):
        if idx < len(signal_matrix):
            signal_scores[i] = np.dot(signal_matrix[idx], jd_spec.signal_weights)
            recency_mults[i] = compute_recency_multiplier(signal_matrix[idx], signal_names)

    # Normalize BM25/RRF scores
    bm25_norm = min_max_norm(bm25_rrf_scores)

    # Honeypot penalties for pool (1.0 = clean, lower = penalized, 0.0 = removed)
    hp_penalties = np.ones(len(pool_indices), dtype=np.float32)
    for i, idx in enumerate(pool_indices):
        if idx < len(honeypot_penalties):
            hp_penalties[i] = float(honeypot_penalties[idx])

    # Skill match bonus
    skill_match = compute_skill_match_bonus(pool_df, jd_spec)

    # Composite score
    composite = (
        W_CE * ce_norm
        + W_SIGNAL * signal_scores
        + W_BM25 * bm25_norm
    )
    
    # Add skill match as a bonus
    composite = 0.70 * composite + 0.30 * skill_match

    # Apply recency multiplier
    composite = composite * recency_mults

    # Apply honeypot penalty multiplier
    composite = composite * hp_penalties
    
    # Apply disqualifier penalty
    disq_penalties = np.ones(len(pool_indices), dtype=np.float32)
    for i, (_, row) in enumerate(pool_df.iterrows()):
        company = str(row.get("current_company", "")).lower()
        if company and any(d in company for d in jd_spec.disqualifier_companies):
            disq_penalties[i] = 0.1
    composite = composite * disq_penalties

    # Build result DataFrame
    result = pd.DataFrame({
        "array_index": pool_indices,
        "candidate_id": pool_df["candidate_id"].values,
        "composite_score": composite,
        "ce_score": ce_norm,
        "signal_score": signal_scores,
        "bm25_score": bm25_norm,
        "skill_match": skill_match,
        "is_honeypot": hp_penalties < 1.0,
        "hp_penalty": hp_penalties,
    })

    # Sort descending
    result = result.sort_values("composite_score", ascending=False).reset_index(drop=True)

    print(f"[fusion] Score stats:")
    print(f"  CE norm: [{ce_norm.min():.4f}, {ce_norm.max():.4f}]")
    print(f"  Signal: [{signal_scores.min():.4f}, {signal_scores.max():.4f}]")
    print(f"  BM25 norm: [{bm25_norm.min():.4f}, {bm25_norm.max():.4f}]")
    print(f"  Skill match: [{skill_match.min():.4f}, {skill_match.max():.4f}]")
    print(f"  Honeypots in pool: {int((hp_penalties < 1.0).sum())}")
    print(f"  Composite: [{composite.min():.4f}, {composite.max():.4f}]")

    return result


def enforce_monotonicity(scores: np.ndarray) -> np.ndarray:
    """
    Enforce strictly non-increasing scores.
    Add tiny epsilon adjustments to break ties while maintaining order.
    Also handle tie-breaking for the validator.
    """
    scores = scores.copy()
    for i in range(1, len(scores)):
        if scores[i] >= scores[i - 1]:
            scores[i] = scores[i - 1] - 1e-9
    return scores
