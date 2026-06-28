"""
preprocess.py
Parses candidates.jsonl into a clean enriched DataFrame.
Writes artifacts/candidates_df.parquet

Handles the nested JSON structure from candidate_schema.json:
  - profile.* fields (anonymized_name, headline, summary, location, etc.)
  - career_history[] array of work experience
  - education[] array
  - skills[] array of {name, proficiency, endorsements, duration_months}
  - redrob_signals.* (23 behavioral signals)
"""

import json
import sys
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (
    load_candidates_lazy,
    ensure_artifacts_dir,
    set_all_seeds,
    title_to_seniority,
    truncate_to_tokens,
    ARTIFACTS_DIR,
)


# ──────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────

def parse_date(date_str: str | None) -> date | None:
    """Parse a date string (YYYY-MM-DD) into a date object."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def compute_yoe_from_history(career_history: list[dict]) -> float:
    """
    Compute years of experience from career history by summing
    non-overlapping tenure spans.
    """
    if not career_history:
        return 0.0

    today = date.today()
    intervals = []

    for job in career_history:
        start = parse_date(job.get("start_date"))
        if not start:
            # Fallback: use duration_months if available
            dur = job.get("duration_months", 0)
            if dur > 0:
                intervals.append((0, dur))
            continue

        if job.get("is_current", False) or not job.get("end_date"):
            end = today
        else:
            end = parse_date(job.get("end_date"))
            if not end:
                end = today

        if end < start:
            continue

        months = (end.year - start.year) * 12 + (end.month - start.month)
        intervals.append((start.toordinal(), end.toordinal()))

    if not intervals:
        return 0.0

    # If we have ordinal intervals, merge overlapping
    if intervals[0][0] > 100:  # ordinal dates
        intervals.sort()
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        total_days = sum(end - start for start, end in merged)
        return total_days / 365.25
    else:
        # Fallback: duration_months based
        return sum(dur for _, dur in intervals) / 12.0


def compute_career_trajectory(career_history: list[dict]) -> float:
    """
    Compute career trajectory slope from title seniority over time.
    Positive = ascending, 0 = flat, negative = declining.
    """
    if not career_history or len(career_history) < 2:
        return 0.0

    points = []
    for job in career_history:
        start = parse_date(job.get("start_date"))
        if not start:
            continue
        tier = title_to_seniority(job.get("title", ""))
        years_from_start = 0  # placeholder
        points.append((start.toordinal(), tier))

    if len(points) < 2:
        return 0.0

    # Sort by date
    points.sort(key=lambda x: x[0])
    base = points[0][0]
    x = np.array([(p[0] - base) / 365.25 for p in points])
    y = np.array([p[1] for p in points])

    if np.std(x) == 0:
        return 0.0

    slope, _, _, _, _ = stats.linregress(x, y)
    return float(slope)


def build_profile_text(record: dict) -> str:
    """
    Build concatenated profile text for BM25 and embedding.
    Combines title, company, skills, education, summary, and career descriptions.
    Kept under ~512 tokens.
    """
    profile = record.get("profile", {})
    parts = []

    # Title and company
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    if title:
        parts.append(title)
    if company:
        parts.append(f"at {company}")

    # Headline
    headline = profile.get("headline", "")
    if headline:
        parts.append(headline)

    # Skills (names only)
    skills = record.get("skills", [])
    skill_names = [s.get("name", "") for s in skills if s.get("name")]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names))

    # Education
    education = record.get("education", [])
    for edu in education:
        degree = edu.get("degree", "")
        field = edu.get("field_of_study", "")
        institution = edu.get("institution", "")
        if degree or field:
            parts.append(f"{degree} {field} {institution}".strip())

    # Summary
    summary = profile.get("summary", "")
    if summary:
        parts.append(summary)

    # Career history titles and descriptions (abbreviated)
    career = record.get("career_history", [])
    for job in career[:3]:  # top 3 roles
        job_title = job.get("title", "")
        job_company = job.get("company", "")
        desc = job.get("description", "")
        if job_title:
            parts.append(f"{job_title} at {job_company}")
        if desc:
            # Take first 100 chars of description
            parts.append(desc[:150])

    # Industry
    industry = profile.get("current_industry", "")
    if industry:
        parts.append(industry)

    text = " ".join(parts)
    return truncate_to_tokens(text, max_tokens=512)


def extract_signal_scalars(signals: dict) -> dict:
    """
    Extract the 23 redrob signals into scalar values for DataFrame columns.
    Handles special types: dates → recency, dicts → aggregates, objects → scalars.
    """
    today = date.today()
    result = {}

    # Direct numeric signals
    result["signal_profile_completeness"] = signals.get("profile_completeness_score", 0.0)
    result["signal_profile_views_30d"] = signals.get("profile_views_received_30d", 0)
    result["signal_applications_30d"] = signals.get("applications_submitted_30d", 0)
    result["signal_recruiter_response_rate"] = signals.get("recruiter_response_rate", 0.0)
    result["signal_avg_response_time_hrs"] = signals.get("avg_response_time_hours", 0.0)
    result["signal_connection_count"] = signals.get("connection_count", 0)
    result["signal_endorsements_received"] = signals.get("endorsements_received", 0)
    result["signal_notice_period_days"] = signals.get("notice_period_days", 0)
    result["signal_github_activity"] = max(signals.get("github_activity_score", -1), 0.0)  # -1 → 0
    result["signal_search_appearance_30d"] = signals.get("search_appearance_30d", 0)
    result["signal_saved_by_recruiters_30d"] = signals.get("saved_by_recruiters_30d", 0)
    result["signal_interview_completion_rate"] = signals.get("interview_completion_rate", 0.0)
    result["signal_offer_acceptance_rate"] = max(signals.get("offer_acceptance_rate", -1), 0.0)  # -1 → 0

    # Boolean signals → 0/1
    result["signal_open_to_work"] = int(signals.get("open_to_work_flag", False))
    result["signal_willing_to_relocate"] = int(signals.get("willing_to_relocate", False))
    result["signal_verified_email"] = int(signals.get("verified_email", False))
    result["signal_verified_phone"] = int(signals.get("verified_phone", False))
    result["signal_linkedin_connected"] = int(signals.get("linkedin_connected", False))

    # Date signals → recency (days since, capped)
    signup = parse_date(signals.get("signup_date"))
    last_active = parse_date(signals.get("last_active_date"))
    result["signal_days_since_signup"] = (today - signup).days if signup else 365
    result["signal_days_since_active"] = (today - last_active).days if last_active else 365

    # Salary midpoint
    salary = signals.get("expected_salary_range_inr_lpa", {})
    sal_min = salary.get("min", 0) if isinstance(salary, dict) else 0
    sal_max = salary.get("max", 0) if isinstance(salary, dict) else 0
    result["signal_salary_midpoint_lpa"] = (sal_min + sal_max) / 2.0 if (sal_min + sal_max) > 0 else 0.0

    # Skill assessment mean score
    assessments = signals.get("skill_assessment_scores", {})
    if isinstance(assessments, dict) and assessments:
        result["signal_skill_assessment_mean"] = sum(assessments.values()) / len(assessments)
    else:
        result["signal_skill_assessment_mean"] = 0.0

    # Work mode → ordinal (remote-friendliness)
    work_mode = signals.get("preferred_work_mode", "onsite")
    mode_map = {"remote": 3, "flexible": 2, "hybrid": 1, "onsite": 0}
    result["signal_work_mode_score"] = mode_map.get(work_mode, 0)

    return result


# ──────────────────────────────────────────────
# Main preprocessing function
# ──────────────────────────────────────────────

def preprocess_candidates(candidates_path: str) -> pd.DataFrame:
    """
    Load candidates.jsonl, parse every record, compute derived features.
    Returns enriched DataFrame.
    """
    set_all_seeds(42)
    print(f"[preprocess] Loading candidates from {candidates_path}...")

    rows = []
    for record in tqdm(load_candidates_lazy(candidates_path), desc="Parsing candidates"):
        profile = record.get("profile", {})
        career = record.get("career_history", [])
        education = record.get("education", [])
        skills_list = record.get("skills", [])
        signals = record.get("redrob_signals", {})
        certifications = record.get("certifications", [])
        languages = record.get("languages", [])

        row = {}

        # ── Top-level ID ──
        row["candidate_id"] = record.get("candidate_id", "")

        # ── Profile fields ──
        row["full_name"] = profile.get("anonymized_name", "")
        row["headline"] = profile.get("headline", "")
        row["summary"] = profile.get("summary", "")
        row["location_city"] = profile.get("location", "")
        row["location_country"] = profile.get("country", "")
        row["years_of_experience"] = float(profile.get("years_of_experience", 0))
        row["current_title"] = profile.get("current_title", "")
        row["current_company"] = profile.get("current_company", "")
        row["current_company_size"] = profile.get("current_company_size", "")
        row["current_industry"] = profile.get("current_industry", "")

        # ── Skills ──
        skill_names = [s.get("name", "") for s in skills_list if s.get("name")]
        row["skills"] = json.dumps(skill_names)
        row["skill_count"] = len(skill_names)

        # Skills detail for honeypot detection
        row["skills_detail"] = json.dumps(skills_list)

        # ── Education ──
        if education:
            highest = max(education, key=lambda e: e.get("end_year", 0))
            row["education_degree"] = highest.get("degree", "")
            row["education_field"] = highest.get("field_of_study", "")
            row["education_institution"] = highest.get("institution", "")
            row["education_tier"] = highest.get("tier", "unknown")
        else:
            row["education_degree"] = ""
            row["education_field"] = ""
            row["education_institution"] = ""
            row["education_tier"] = "unknown"

        # ── Career history (JSON for later use) ──
        row["career_history_json"] = json.dumps(career)

        # ── Languages ──
        lang_list = [l.get("language", "") for l in languages if l.get("language")]
        row["languages"] = json.dumps(lang_list)

        # ── Certifications count ──
        row["certification_count"] = len(certifications)

        # ── Redrob signals (raw for later) ──
        row["notice_period_days"] = signals.get("notice_period_days", 0)

        salary = signals.get("expected_salary_range_inr_lpa", {})
        row["expected_ctc_min"] = salary.get("min", 0) if isinstance(salary, dict) else 0
        row["expected_ctc_max"] = salary.get("max", 0) if isinstance(salary, dict) else 0

        row["preferred_work_mode"] = signals.get("preferred_work_mode", "onsite")
        row["open_to_work"] = signals.get("open_to_work_flag", False)
        row["willing_to_relocate"] = signals.get("willing_to_relocate", False)

        # ── Signal scalars (23 signals normalized) ──
        signal_vals = extract_signal_scalars(signals)
        row.update(signal_vals)

        # ── Derived features ──

        # 1. Computed YOE
        computed_yoe = compute_yoe_from_history(career)
        row["computed_yoe_years"] = round(computed_yoe, 2)
        row["yoe_discrepancy"] = round(abs(computed_yoe - row["years_of_experience"]), 2)

        # 2. Career trajectory
        row["career_trajectory"] = round(compute_career_trajectory(career), 4)

        # 3. Profile completeness (fraction of optional fields that are non-null)
        optional_fields_present = sum([
            bool(profile.get("headline")),
            bool(profile.get("summary")),
            len(skills_list) > 0,
            len(education) > 0,
            len(certifications) > 0,
            len(languages) > 0,
            bool(profile.get("current_industry")),
            bool(profile.get("current_company_size")),
            signals.get("github_activity_score", -1) >= 0,
            signals.get("linkedin_connected", False),
        ])
        row["profile_completeness"] = round(optional_fields_present / 10.0, 2)

        # 4. Has portfolio links
        row["has_github"] = signals.get("github_activity_score", -1) >= 0
        row["has_linkedin"] = signals.get("linkedin_connected", False)

        # 5. Profile text for BM25 and embedding
        row["profile_text"] = build_profile_text(record)

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"[preprocess] Parsed {len(df)} candidates")
    print(f"[preprocess] Columns: {list(df.columns)}")
    return df


def save_parquet(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """Save DataFrame to parquet."""
    ensure_artifacts_dir()
    if output_path is None:
        output_path = ARTIFACTS_DIR / "candidates_df.parquet"
    df.to_parquet(output_path, index=False, engine="pyarrow")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[preprocess] Saved {output_path} ({size_mb:.1f} MB, {len(df)} rows)")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess candidates into parquet")
    parser.add_argument("--candidates", type=str, required=True,
                        help="Path to candidates.jsonl or candidates.jsonl.gz")
    args = parser.parse_args()

    df = preprocess_candidates(args.candidates)
    save_parquet(df)
    print("[preprocess] Done!")
