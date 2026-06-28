"""
honeypot.py
Rule-based honeypot detection with graduated confidence scoring.

Implements 6 HARD rules (one trigger = flagged, penalty_multiplier=0.0) and
6 SOFT rules (1 trigger = mild flag, 2+ = strong flag).

Candidates are penalized but NOT removed from the pool.
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import ensure_artifacts_dir, ARTIFACTS_DIR


@dataclass
class HoneypotResult:
    candidate_id: str
    is_flagged: bool
    confidence: float          # 0.0 = clean, 1.0 = certain honeypot
    triggered_rules: list[str] = field(default_factory=list)
    penalty_multiplier: float = 1.0  # 1.0 = no penalty, 0.0 = score zeroed


def parse_date_safe(date_str: str | None) -> date | None:
    """Parse date string safely."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


# ═══════════════════════════════════════════════
# HARD RULES — any single trigger = flagged
# ═══════════════════════════════════════════════

def check_hard_2_skill_contradiction(skills_detail_json: str) -> bool:
    """
    HARD-2: Skill/experience contradiction.
    Count skills where proficiency == "expert" AND duration_months == 0.
    If count >= 3: HARD flag.
    """
    try:
        skills = json.loads(skills_detail_json) if isinstance(skills_detail_json, str) else skills_detail_json
    except (json.JSONDecodeError, TypeError):
        return False

    if not skills:
        return False

    expert_zero = 0
    for s in skills:
        prof = s.get("proficiency", "").lower()
        dur = s.get("duration_months", 1)
        if prof == "expert" and dur == 0:
            expert_zero += 1

    return expert_zero >= 3


def check_hard_3_timeline_overlap(career_json: str) -> bool:
    """
    HARD-3: Timeline overlap.
    Check all work experience entries for date overlaps > 6 months.
    Two full-time roles simultaneously for >6 months: HARD flag.
    """
    try:
        career = json.loads(career_json) if isinstance(career_json, str) else career_json
    except (json.JSONDecodeError, TypeError):
        return False

    if not career or len(career) < 2:
        return False

    # Build intervals
    intervals = []
    for job in career:
        start = parse_date_safe(job.get("start_date"))
        if not start:
            continue
        if job.get("is_current", False) or not job.get("end_date"):
            end = date.today()
        else:
            end = parse_date_safe(job.get("end_date"))
            if not end:
                continue
        if end >= start:
            intervals.append((start, end))

    # Check all pairs for >6 month overlap
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            s1, e1 = intervals[i]
            s2, e2 = intervals[j]
            overlap_start = max(s1, s2)
            overlap_end = min(e1, e2)
            if overlap_end > overlap_start:
                overlap_months = (overlap_end.year - overlap_start.year) * 12 + \
                                 (overlap_end.month - overlap_start.month)
                if overlap_months > 6:
                    return True

    return False


def check_hard_4_yoe_impossibility(yoe_self: float) -> bool:
    """
    HARD-4: YOE impossibility.
    If self-reported years_of_experience > 45: HARD flag.
    """
    return yoe_self > 45


def check_hard_5_education_timeline(row: pd.Series) -> bool:
    """
    HARD-5: Education timeline impossibility.
    If candidate claims PhD + computed_yoe < 3 but self-reported YOE > 15:
    likely fabricated timeline.
    Also: if education_degree contains PhD and YOE > 30 and computed_yoe < 5.
    """
    edu = str(row.get("education_degree", "")).lower()
    yoe_self = float(row.get("years_of_experience", 0))
    computed_yoe = float(row.get("computed_yoe_years", 0))

    if "phd" in edu or "ph.d" in edu or "doctorate" in edu:
        # PhD + claims >15 yrs but computed shows <3: impossible timeline
        if yoe_self > 15 and computed_yoe < 3:
            return True
        # PhD + claims >30 yrs but computed shows <5
        if yoe_self > 30 and computed_yoe < 5:
            return True

    return False


def check_hard_6_skill_proficiency(skills_detail_json: str) -> bool:
    """
    HARD-6: Skill proficiency contradiction (relaxed).
    For each skill: if proficiency == "expert" AND duration_months < 1 (essentially 0):
      increment expert_zero_count
    If proficiency == "advanced" AND duration_months < 6:
      increment advanced_low_count
    If expert_zero >= 2 OR (expert_zero >= 1 AND advanced_low >= 2): HARD flag
    """
    try:
        skills = json.loads(skills_detail_json) if isinstance(skills_detail_json, str) else skills_detail_json
    except (json.JSONDecodeError, TypeError):
        return False

    if not skills:
        return False

    expert_zero = 0
    advanced_low = 0
    for s in skills:
        prof = s.get("proficiency", "").lower()
        dur = s.get("duration_months", -1)
        if dur < 0:
            continue  # No duration data, skip
        if prof == "expert" and dur < 1:
            expert_zero += 1
        elif prof == "advanced" and dur < 6:
            advanced_low += 1

    return expert_zero >= 2 or (expert_zero >= 1 and advanced_low >= 2)


# ═══════════════════════════════════════════════
# SOFT RULES — graduated penalty
# ═══════════════════════════════════════════════

def check_soft_1_skill_stuffing(skill_count: int, profile_completeness: float) -> bool:
    """SOFT-1: Skill stuffing. skill_count > 50 AND profile_completeness > 0.95."""
    return skill_count > 50 and profile_completeness > 0.95


def check_soft_2_signal_envelope(row_signals: pd.Series, percentile_999: pd.Series) -> bool:
    """SOFT-2: Signal envelope violation. >4 signals in 99.9th percentile."""
    outlier_count = (row_signals > percentile_999).sum()
    return outlier_count > 4


def check_soft_3_yoe_discrepancy(yoe_discrepancy: float) -> bool:
    """SOFT-3: YOE discrepancy > 3.0 years (self-reported vs computed)."""
    return yoe_discrepancy > 3.0


def check_soft_4_uniform_signals(row_signals: pd.Series) -> bool:
    """SOFT-4: Suspiciously uniform signals (std < 0.05)."""
    return row_signals.std() < 0.05


def check_soft_5_company_repetition(career_json: str) -> bool:
    """
    SOFT-5: Same company appears 3+ times in work history with non-overlapping dates.
    Legitimate re-hires exist but 3+ is suspicious.
    """
    try:
        career = json.loads(career_json) if isinstance(career_json, str) else career_json
    except (json.JSONDecodeError, TypeError):
        return False

    if not career or len(career) < 3:
        return False

    # Group jobs by company
    company_jobs: dict[str, list] = {}
    for job in career:
        company = job.get("company", "").strip().lower()
        if company:
            company_jobs.setdefault(company, []).append(job)

    # Check for 4+ distinct occurrences (ignoring date overlap for simplicity but raising threshold)
    # Raising threshold to 4 to prevent normal promotion tracks from being flagged
    return any(len(jobs) >= 4 for jobs in company_jobs.values())


def check_soft_6_round_numbers(row: pd.Series, signal_cols: list[str]) -> bool:
    """
    SOFT-6: Suspiciously round numbers across numeric fields.
    If >60% of numeric signal fields are exact integers AND skill_count > 30: flag.
    """
    skill_count = int(row.get("skill_count", 0))
    if skill_count <= 30:
        return False

    round_count = 0
    total_count = 0

    for col in signal_cols:
        val = float(row.get(col, 0))
        total_count += 1
        if val == int(val):  # Exact integer
            round_count += 1

    # Also check YOE
    yoe = float(row.get("years_of_experience", 0))
    total_count += 1
    if yoe == int(yoe):
        round_count += 1

    if total_count == 0:
        return False

    return (round_count / total_count) > 0.60


# ═══════════════════════════════════════════════
# MAIN DETECTION
# ═══════════════════════════════════════════════

def detect_honeypots(df: pd.DataFrame) -> dict[str, HoneypotResult]:
    """
    Main honeypot detection function.
    Returns dict mapping candidate_id -> HoneypotResult.
    Uses graduated confidence and penalty multipliers:
      HARD rule → confidence=1.0, penalty_multiplier=0.0
      2+ SOFT  → confidence=0.8, penalty_multiplier=0.05
      1 SOFT   → confidence=0.4, penalty_multiplier=0.3
    """
    n = len(df)
    results: dict[str, HoneypotResult] = {}

    # Pre-compute signal columns and percentiles for SOFT-2
    signal_cols = [c for c in df.columns if c.startswith("signal_")
                   and df[c].dtype in [np.float64, np.int64, float, int]]
    signal_df = df[signal_cols].astype(float) if signal_cols else pd.DataFrame()
    percentile_999 = signal_df.quantile(0.999) if len(signal_df) > 0 else pd.Series()

    # Track counts for logging
    hard_counts = {"HARD-2": 0, "HARD-3": 0, "HARD-4": 0, "HARD-5": 0, "HARD-6": 0}
    soft_counts = {"SOFT-1": 0, "SOFT-2": 0, "SOFT-3": 0, "SOFT-4": 0, "SOFT-5": 0, "SOFT-6": 0}

    print("[honeypot] Running detection on {} candidates...".format(n))

    for idx in range(n):
        row = df.iloc[idx]
        cid = str(row.get("candidate_id", f"IDX_{idx}"))
        triggered = []

        # ── HARD rules ──
        skills_detail = row.get("skills_detail", "[]")
        career_json = row.get("career_history_json", "[]")

        if check_hard_2_skill_contradiction(skills_detail):
            triggered.append("HARD-2")
            hard_counts["HARD-2"] += 1

        if check_hard_3_timeline_overlap(career_json):
            triggered.append("HARD-3")
            hard_counts["HARD-3"] += 1

        if check_hard_4_yoe_impossibility(float(row.get("years_of_experience", 0))):
            triggered.append("HARD-4")
            hard_counts["HARD-4"] += 1

        if check_hard_5_education_timeline(row):
            triggered.append("HARD-5")
            hard_counts["HARD-5"] += 1

        if check_hard_6_skill_proficiency(skills_detail):
            triggered.append("HARD-6")
            hard_counts["HARD-6"] += 1

        # ── SOFT rules ──
        soft_triggered = []

        if check_soft_1_skill_stuffing(
            int(row.get("skill_count", 0)),
            float(row.get("profile_completeness", 0))
        ):
            soft_triggered.append("SOFT-1")
            soft_counts["SOFT-1"] += 1

        if len(signal_df) > 0:
            if check_soft_2_signal_envelope(signal_df.iloc[idx], percentile_999):
                soft_triggered.append("SOFT-2")
                soft_counts["SOFT-2"] += 1

            if check_soft_4_uniform_signals(signal_df.iloc[idx]):
                soft_triggered.append("SOFT-4")
                soft_counts["SOFT-4"] += 1

        if check_soft_3_yoe_discrepancy(float(row.get("yoe_discrepancy", 0))):
            soft_triggered.append("SOFT-3")
            soft_counts["SOFT-3"] += 1

        if check_soft_5_company_repetition(career_json):
            soft_triggered.append("SOFT-5")
            soft_counts["SOFT-5"] += 1

        if check_soft_6_round_numbers(row, signal_cols):
            soft_triggered.append("SOFT-6")
            soft_counts["SOFT-6"] += 1

        triggered.extend(soft_triggered)

        # ── Determine confidence and penalty ──
        has_hard = any(r.startswith("HARD") for r in triggered)
        n_soft = len(soft_triggered)

        if has_hard:
            confidence = 1.0
            penalty_multiplier = 0.0  # Effectively removed
            is_flagged = True
        elif n_soft >= 2:
            confidence = 0.8
            penalty_multiplier = 0.05  # Nearly zeroed
            is_flagged = True
        elif n_soft == 1:
            confidence = 0.4
            penalty_multiplier = 0.3  # Significant penalty
            is_flagged = True
        else:
            confidence = 0.0
            penalty_multiplier = 1.0  # No penalty
            is_flagged = False

        results[cid] = HoneypotResult(
            candidate_id=cid,
            is_flagged=is_flagged,
            confidence=confidence,
            triggered_rules=triggered,
            penalty_multiplier=penalty_multiplier,
        )

    # ── Print summary ──
    flagged = sum(1 for r in results.values() if r.is_flagged)
    hard_only = sum(1 for r in results.values()
                    if r.is_flagged and any(t.startswith("HARD") for t in r.triggered_rules))
    soft_only = flagged - hard_only

    print("[honeypot] HARD rule results:")
    for rule, count in hard_counts.items():
        print(f"  {rule}: {count} flagged")
    print("[honeypot] SOFT rule results:")
    for rule, count in soft_counts.items():
        print(f"  {rule}: {count} triggered")
    print(f"[honeypot] Total flagged: {flagged} / {n} ({flagged/n*100:.2f}%)")
    print(f"  HARD-flagged: {hard_only}, SOFT-only flagged: {soft_only}")

    return results


def audit_honeypot_flags(df: pd.DataFrame, results: dict[str, HoneypotResult]) -> None:
    """
    Print a detailed audit of honeypot detection results.
    Warns if count is outside expected range (60-200).
    """
    flagged = [r for r in results.values() if r.is_flagged]
    flagged_sorted = sorted(flagged, key=lambda r: -r.confidence)

    # Breakdown by confidence tier
    tier_high = sum(1 for r in flagged if r.confidence >= 0.8)
    tier_mid = sum(1 for r in flagged if 0.3 < r.confidence < 0.8)
    tier_low = sum(1 for r in flagged if r.confidence <= 0.3)

    print(f"\n{'='*60}")
    print(f"  HONEYPOT AUDIT")
    print(f"{'='*60}")
    print(f"  Total flagged: {len(flagged)}")
    print(f"  High confidence (>=0.8): {tier_high}")
    print(f"  Medium confidence (0.3-0.8): {tier_mid}")
    print(f"  Low confidence (<=0.3): {tier_low}")

    print(f"\n  Top 20 flagged candidates:")
    for i, r in enumerate(flagged_sorted[:20]):
        # Get candidate data
        mask = df["candidate_id"] == r.candidate_id
        if mask.any():
            row = df[mask].iloc[0]
            yoe_self = row.get("years_of_experience", 0)
            computed_yoe = row.get("computed_yoe_years", 0)
            skill_count = row.get("skill_count", 0)
        else:
            yoe_self = computed_yoe = skill_count = "?"

        print(f"    {i+1:2d}. {r.candidate_id} | conf={r.confidence:.1f} | "
              f"rules={r.triggered_rules} | "
              f"YOE={yoe_self}/{computed_yoe} | skills={skill_count}")

    # Warnings
    if len(flagged) < 60:
        print(f"\n  ALERT: Honeypot detection below expected range ({len(flagged)} < 60).")
        print(f"  Tighten SOFT rule thresholds or add rules before submitting.")
    elif len(flagged) > 200:
        print(f"\n  ALERT: Over-flagging detected ({len(flagged)} > 200).")
        print(f"  Check thresholds — too many false positives will hurt NDCG.")
    else:
        print(f"\n  [OK] Flagged count ({len(flagged)}) is within expected range (60-200).")

    print(f"{'='*60}\n")


def check_top100_for_honeypots(
    top100_df: pd.DataFrame,
    honeypot_results: dict[str, HoneypotResult],
) -> None:
    """
    Check if any flagged honeypots are in the top 100.
    Prints warnings. sys.exit(1) if >10 flagged in top 100.
    """
    flagged_in_top = []
    for rank_idx, (_, row) in enumerate(top100_df.iterrows(), start=1):
        cid = row["candidate_id"]
        result = honeypot_results.get(cid)
        if result and result.is_flagged:
            flagged_in_top.append((rank_idx, result))

    if flagged_in_top:
        print(f"\n[HONEYPOT CHECK] WARNING: {len(flagged_in_top)} flagged candidate(s) in top 100:")
        for rank, r in flagged_in_top:
            print(f"  Rank {rank:3d} | {r.candidate_id} | confidence={r.confidence:.1f} | "
                  f"penalty={r.penalty_multiplier:.2f} | rules={r.triggered_rules}")

        if len(flagged_in_top) > 10:
            print("\n  CRITICAL: DISQUALIFICATION RISK. More than 10 honeypots in top 100.")
            print("  Do not submit until honeypot_penalty in score_fusion.py is increased.")
            sys.exit(1)
        else:
            print(f"  Review these manually before submitting.")
    else:
        print(f"\n[HONEYPOT CHECK] [OK] No flagged honeypots in top 100.")


# ═══════════════════════════════════════════════
# SAVE / LOAD HELPERS
# ═══════════════════════════════════════════════

def results_to_flags_array(results: dict[str, HoneypotResult], df: pd.DataFrame) -> np.ndarray:
    """Convert HoneypotResult dict to boolean numpy array aligned with df index."""
    flags = np.zeros(len(df), dtype=bool)
    for idx in range(len(df)):
        cid = str(df.iloc[idx]["candidate_id"])
        result = results.get(cid)
        if result and result.is_flagged:
            flags[idx] = True
    return flags


def results_to_penalty_array(results: dict[str, HoneypotResult], df: pd.DataFrame) -> np.ndarray:
    """Convert HoneypotResult dict to penalty_multiplier numpy array aligned with df index."""
    penalties = np.ones(len(df), dtype=np.float32)  # 1.0 = no penalty
    for idx in range(len(df)):
        cid = str(df.iloc[idx]["candidate_id"])
        result = results.get(cid)
        if result:
            penalties[idx] = result.penalty_multiplier
    return penalties


def save_honeypot_results(
    results: dict[str, HoneypotResult],
    df: pd.DataFrame,
    output_dir: Path | None = None,
) -> None:
    """Save honeypot flags and penalty multipliers to numpy arrays."""
    ensure_artifacts_dir()
    if output_dir is None:
        output_dir = ARTIFACTS_DIR

    flags = results_to_flags_array(results, df)
    penalties = results_to_penalty_array(results, df)

    np.save(output_dir / "honeypot_flags.npy", flags)
    np.save(output_dir / "honeypot_penalties.npy", penalties)

    flagged_count = flags.sum()
    print(f"[honeypot] Saved {output_dir / 'honeypot_flags.npy'} ({flagged_count} flagged out of {len(flags)})")
    print(f"[honeypot] Saved {output_dir / 'honeypot_penalties.npy'}")

    # Also save detailed results as JSON for debugging
    results_json = {}
    for cid, r in results.items():
        if r.is_flagged:
            results_json[cid] = {
                "confidence": r.confidence,
                "penalty_multiplier": r.penalty_multiplier,
                "triggered_rules": r.triggered_rules,
            }
    with open(output_dir / "honeypot_details.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"[honeypot] Saved {output_dir / 'honeypot_details.json'} ({len(results_json)} entries)")


# Legacy compatibility
def save_honeypot_flags(flags: pd.Series, output_path: Path | None = None) -> Path:
    """Save honeypot flags to numpy array (legacy interface)."""
    ensure_artifacts_dir()
    if output_path is None:
        output_path = ARTIFACTS_DIR / "honeypot_flags.npy"
    np.save(output_path, flags.values.astype(bool))
    print(f"[honeypot] Saved {output_path} ({flags.sum()} flagged out of {len(flags)})")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=str(ARTIFACTS_DIR / "candidates_df.parquet"))
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    results = detect_honeypots(df)
    audit_honeypot_flags(df, results)
    save_honeypot_results(results, df)
