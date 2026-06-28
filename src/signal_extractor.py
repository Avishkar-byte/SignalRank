"""
signal_extractor.py
Builds the signal matrix from the parsed dataframe.
Maps each candidate's redrob_signals values to a normalized float array.

Output:
  artifacts/signal_matrix.npy  shape (N, 23) float32
  artifacts/signal_names.npy   shape (23,) string array
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import ensure_artifacts_dir, ARTIFACTS_DIR


# The 23 signal columns in order (matching extract_signal_scalars output)
SIGNAL_COLUMNS = [
    "signal_profile_completeness",
    "signal_profile_views_30d",
    "signal_applications_30d",
    "signal_recruiter_response_rate",
    "signal_avg_response_time_hrs",
    "signal_connection_count",
    "signal_endorsements_received",
    "signal_notice_period_days",
    "signal_github_activity",
    "signal_search_appearance_30d",
    "signal_saved_by_recruiters_30d",
    "signal_interview_completion_rate",
    "signal_offer_acceptance_rate",
    "signal_open_to_work",
    "signal_willing_to_relocate",
    "signal_verified_email",
    "signal_verified_phone",
    "signal_linkedin_connected",
    "signal_days_since_signup",
    "signal_days_since_active",
    "signal_salary_midpoint_lpa",
    "signal_skill_assessment_mean",
    "signal_work_mode_score",
]

# Human-readable signal names (for display)
SIGNAL_NAMES = [
    "profile_completeness_score",
    "profile_views_received_30d",
    "applications_submitted_30d",
    "recruiter_response_rate",
    "avg_response_time_hours",
    "connection_count",
    "endorsements_received",
    "notice_period_days",
    "github_activity_score",
    "search_appearance_30d",
    "saved_by_recruiters_30d",
    "interview_completion_rate",
    "offer_acceptance_rate",
    "open_to_work_flag",
    "willing_to_relocate",
    "verified_email",
    "verified_phone",
    "linkedin_connected",
    "days_since_signup",
    "days_since_active",
    "salary_midpoint_lpa",
    "skill_assessment_mean",
    "work_mode_score",
]


def build_signal_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Build the normalized signal matrix from DataFrame columns.

    For each signal:
      - Extract the column from df
      - Handle nulls: fill with column median
      - Min-max normalize to [0, 1] across the full candidate pool
      - For inverted signals (avg_response_time, notice_period, days_since_active):
        invert so higher = better

    Returns float32 array shape (N, 23).
    """
    print(f"[signals] Building signal matrix from {len(df)} candidates...")

    # Inverted signals: lower is better, so we invert after normalization
    INVERTED = {
        "signal_avg_response_time_hrs",   # lower response time = better
        "signal_notice_period_days",       # shorter notice = better
        "signal_days_since_active",        # more recent = better (lower days)
        "signal_days_since_signup",        # more recent signup isn't necessarily better, but less stale
    }

    matrix = np.zeros((len(df), len(SIGNAL_COLUMNS)), dtype=np.float32)

    for i, col in enumerate(SIGNAL_COLUMNS):
        if col in df.columns:
            values = df[col].astype(float).values.copy()
        else:
            print(f"  WARNING: signal column '{col}' not found, filling with 0")
            values = np.zeros(len(df), dtype=np.float32)

        # Handle NaN: fill with median
        nan_mask = np.isnan(values)
        if nan_mask.any():
            median_val = np.nanmedian(values)
            values[nan_mask] = median_val

        # Min-max normalize to [0, 1]
        vmin = values.min()
        vmax = values.max()
        if vmax > vmin:
            values = (values - vmin) / (vmax - vmin)
        else:
            values = np.zeros_like(values)

        # Invert if needed (so higher = better)
        if col in INVERTED:
            values = 1.0 - values

        matrix[:, i] = values

    print(f"[signals] Signal matrix shape: {matrix.shape}")
    print(f"[signals] Value range: [{matrix.min():.4f}, {matrix.max():.4f}]")
    return matrix


def save_signal_matrix(matrix: np.ndarray) -> None:
    """Save signal matrix and signal names."""
    ensure_artifacts_dir()

    matrix_path = ARTIFACTS_DIR / "signal_matrix.npy"
    names_path = ARTIFACTS_DIR / "signal_names.npy"

    np.save(matrix_path, matrix)
    np.save(names_path, np.array(SIGNAL_NAMES))

    mat_mb = matrix_path.stat().st_size / (1024 * 1024)
    print(f"[signals] Saved {matrix_path} ({mat_mb:.1f} MB)")
    print(f"[signals] Saved {names_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=str(ARTIFACTS_DIR / "candidates_df.parquet"))
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    matrix = build_signal_matrix(df)
    save_signal_matrix(matrix)
    print("[signals] Done!")
