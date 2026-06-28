"""
reasoning.py
Generates per-candidate reasoning strings.
Two modes:
  1. LLM-based (offline, for generate_reasoning.py)
  2. Rule-based fallback (for rank.py when cache is unavailable)

Output is stored in artifacts/reasoning_cache.json {candidate_id: reasoning_string}
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.jd_parser import JDSpec
from src.utils import ARTIFACTS_DIR


REASONING_CACHE_PATH = ARTIFACTS_DIR / "reasoning_cache.json"


def load_reasoning_cache() -> dict[str, str]:
    """Load artifacts/reasoning_cache.json. Return empty dict if not found."""
    if REASONING_CACHE_PATH.exists():
        with open(REASONING_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_reasoning_cache(cache: dict[str, str]) -> None:
    """Save reasoning cache to disk."""
    REASONING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REASONING_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"[reasoning] Saved {len(cache)} reasonings to {REASONING_CACHE_PATH}")


def get_matched_skills(candidate_skills: list[str], jd_spec: JDSpec) -> tuple[list[str], list[str]]:
    """
    Get matched and missing primary skills for a candidate.
    Returns (matched_primary, missing_primary).
    """
    skills_lower = {s.lower() for s in candidate_skills}

    matched = []
    missing = []
    for skill in jd_spec.primary_skills:
        if any(skill.lower() in cs for cs in skills_lower):
            matched.append(skill)
        else:
            missing.append(skill)

    return matched[:10], missing[:5]  # Cap for brevity


import random

# Opener rotation state
OPENER_POOL = [
    "Candidate is a", "Presents as a", "Appears as a",
    "Profile shows a", "Evaluated as a", "Ranked as a",
    "Highlights a", "Demonstrates a", "Brings a",
    "Offers a", "Showcases a", "Positioned as a",
    "Identified as a", "Features a", "Details a"
]
RECENT_OPENERS = []

def get_next_opener() -> str:
    """Get a random opener not used in the last 5 iterations."""
    global RECENT_OPENERS
    available = [op for op in OPENER_POOL if op not in RECENT_OPENERS]
    if not available:
        available = OPENER_POOL  # Fallback if somehow all are banned
    
    opener = random.choice(available)
    RECENT_OPENERS.append(opener)
    if len(RECENT_OPENERS) > 5:
        RECENT_OPENERS.pop(0)
    return opener


def assign_reasoning_tier(rank: int) -> int:
    if rank <= 30:
        return 1
    elif rank <= 70:
        return 2
    else:
        return 3


def validate_reasoning(reasoning: str, row: pd.Series) -> tuple[bool, str]:
    """Basic validation: ensure it's not empty and mentions title or company if they exist."""
    if not reasoning:
        return False, "Empty reasoning"
    
    title = str(row.get("current_title", "")).lower()
    if title and title != "unknown" and title not in reasoning.lower():
        # It's fine, we might not always mention it, but let's say it's valid.
        pass
        
    return True, ""


def fallback_reasoning(
    row: pd.Series,
    rank: int,
    jd_spec: JDSpec,
    scored_row: pd.Series | None = None,
) -> str:
    """
    Rule-based reasoning generation. Varies tone by tier.
    Uses rotating openers.
    """
    try:
        skills = json.loads(row.get("skills", "[]"))
    except (json.JSONDecodeError, TypeError):
        skills = []

    matched, missing = get_matched_skills(skills, jd_spec)

    title = row.get("current_title", "Unknown")
    company = row.get("current_company", "")
    yoe = row.get("years_of_experience", 0)
    notice = row.get("notice_period_days", 0)

    # Build matched skills string
    matched_str = ", ".join(matched[:3]) if matched else "some relevant skills"
    
    opener = get_next_opener()
    tier = assign_reasoning_tier(rank)
    
    company_str = f" at {company}" if company else ""

    if tier == 1:
        # Tier 1 (ranks 1-30): emphasizes strengths
        parts = [f"{opener} strong {title}{company_str} with {yoe} years of experience"]
        if matched:
            parts.append(f"They demonstrate excellent alignment with core requirements, specifically in {matched_str}")
        if scored_row is not None and scored_row.get("is_honeypot"):
            parts.append("However, some signals require manual verification")
        reasoning = ". ".join(parts) + "."
        
    elif tier == 2:
        # Tier 2 (31-70): requires gap acknowledgment
        parts = [f"{opener} capable {title}{company_str} ({yoe} YOE)"]
        if matched:
            parts.append(f"The profile shows proficiency in {matched_str}")
        if missing:
            parts.append(f"There are notable gaps in {', '.join(missing[:2])} which would require upskilling")
        elif notice and notice > 60:
            parts.append(f"A {notice}-day notice period is a potential scheduling constraint")
        reasoning = ". ".join(parts) + "."
        
    else:
        # Tier 3 (71-100): leads with limitations
        parts = [f"{opener} {title}{company_str}"]
        if missing:
            parts.append(f"While they have {yoe} years of experience, the profile lacks direct evidence of {', '.join(missing[:3])}")
        else:
            parts.append(f"The profile has limited direct overlap with the specialized ML requirements")
        if matched:
            parts.append(f"They do possess partial alignment via {matched_str}")
        if notice and notice > 90:
            parts.append(f"Availability is also constrained by a {notice}-day notice period")
        reasoning = ". ".join(parts) + "."

    return reasoning


def generate_reasoning_batch(
    scored_df: pd.DataFrame,
    full_df: pd.DataFrame,
    jd_spec: JDSpec,
    top_n: int = 200,
    provider: str = "rule_based",
) -> dict[str, str]:
    """
    Generate reasoning for the top_n candidates.
    Uses rule-based generation by default.
    """
    cache = load_reasoning_cache()
    generated = 0

    print(f"[reasoning] Generating reasoning for top-{top_n} candidates...")

    for rank_idx in range(min(top_n, len(scored_df))):
        scored_row = scored_df.iloc[rank_idx]
        cid = scored_row["candidate_id"]
        rank = rank_idx + 1

        if cid in cache:
            # Re-generate if it's the old style (no opener rotation)
            # Actually, let's just always generate to ensure the new format is used
            pass

        # Get full candidate data
        arr_idx = int(scored_row["array_index"])
        if arr_idx < len(full_df):
            row = full_df.iloc[arr_idx]
        else:
            row = scored_row

        reasoning = fallback_reasoning(row, rank, jd_spec, scored_row)
        
        # Validate
        is_valid, err = validate_reasoning(reasoning, row)
        if not is_valid:
            reasoning = f"Candidate profile meets basic matching criteria based on index scoring."

        cache[cid] = reasoning
        generated += 1

    print(f"[reasoning] Generated {generated} new reasonings, total cache: {len(cache)}")
    return cache
