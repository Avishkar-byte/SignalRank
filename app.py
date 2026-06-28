"""
app.py
HuggingFace Spaces Gradio sandbox demo.
Accepts a JSON file of up to 100 candidates, runs a mini ranking pipeline,
and outputs a ranked CSV download.
"""

import json
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import set_all_seeds, truncate_to_tokens, tokenize_text
from src.preprocess import (
    compute_yoe_from_history,
    compute_career_trajectory,
    extract_signal_scalars,
    build_profile_text,
)
from src.jd_parser import parse_jd
from src.signal_extractor import SIGNAL_NAMES, build_signal_matrix
from src.rerank import build_candidate_text
from src.score_fusion import min_max_norm, enforce_monotonicity
from src.reasoning import fallback_reasoning


# Default JD text (embedded for sandbox use)
DEFAULT_JD = """Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid)
Experience Required: 5-9 years

Required skills: Production experience with embeddings-based retrieval systems,
vector databases (FAISS, Pinecone, etc.), strong Python, evaluation frameworks
for ranking systems (NDCG, MRR, MAP).

Nice to have: LLM fine-tuning (LoRA, QLoRA), learning-to-rank, HR-tech exposure,
distributed systems, open-source contributions.

Looking for someone with deep technical depth in modern ML systems —
embeddings, retrieval, ranking, LLMs, fine-tuning — combined with a
scrappy product-engineering attitude."""


def preprocess_candidates_from_list(records: list[dict]) -> pd.DataFrame:
    """Simplified preprocessing for small candidate lists (sandbox mode)."""
    rows = []
    for record in records:
        profile = record.get("profile", {})
        career = record.get("career_history", [])
        education = record.get("education", [])
        skills_list = record.get("skills", [])
        signals = record.get("redrob_signals", {})

        row = {}
        row["candidate_id"] = record.get("candidate_id", "")
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

        skill_names = [s.get("name", "") for s in skills_list if s.get("name")]
        row["skills"] = json.dumps(skill_names)
        row["skill_count"] = len(skill_names)
        row["skills_detail"] = json.dumps(skills_list)

        if education:
            highest = max(education, key=lambda e: e.get("end_year", 0))
            row["education_degree"] = highest.get("degree", "")
            row["education_field"] = highest.get("field_of_study", "")
        else:
            row["education_degree"] = ""
            row["education_field"] = ""

        row["notice_period_days"] = signals.get("notice_period_days", 0)
        row["career_history_json"] = json.dumps(career)

        signal_vals = extract_signal_scalars(signals)
        row.update(signal_vals)

        row["computed_yoe_years"] = compute_yoe_from_history(career)
        row["yoe_discrepancy"] = abs(row["computed_yoe_years"] - row["years_of_experience"])
        row["career_trajectory"] = compute_career_trajectory(career)
        row["profile_completeness"] = 0.5
        row["profile_text"] = build_profile_text(record)

        rows.append(row)

    return pd.DataFrame(rows)


def rank_candidates(candidates_file, jd_text: str) -> tuple[pd.DataFrame, str]:
    """
    Mini ranking pipeline for sandbox demo.
    Skips FAISS/BM25 (too small), uses cross-encoder directly.
    """
    set_all_seeds(42)
    t_start = time.time()
    status_lines = []

    # Parse candidates
    if candidates_file is None:
        return pd.DataFrame(), "Please upload a candidates JSON file."

    try:
        with open(candidates_file.name, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("["):
                records = json.loads(content)
            else:
                records = [json.loads(line) for line in content.split("\n") if line.strip()]
    except Exception as e:
        return pd.DataFrame(), f"Error parsing file: {e}"

    if len(records) > 100:
        records = records[:100]
        status_lines.append("Truncated to 100 candidates (max for sandbox)")

    status_lines.append(f"Loaded {len(records)} candidates")

    # Preprocess
    df = preprocess_candidates_from_list(records)
    status_lines.append(f"Preprocessed {len(df)} candidates")

    # Parse JD
    if not jd_text or not jd_text.strip():
        jd_text = DEFAULT_JD

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp:
        tmp.write(jd_text)
        tmp_path = tmp.name

    try:
        jd_spec = parse_jd(tmp_path, SIGNAL_NAMES)
    except Exception as e:
        return pd.DataFrame(), f"Error parsing JD: {e}"

    # Build signal matrix
    signal_mat = build_signal_matrix(df)

    # Cross-encoder scoring
    try:
        from sentence_transformers import CrossEncoder
        from src.rerank import CE_MODEL

        ce_model = CrossEncoder(CE_MODEL, max_length=512, device="cpu")
        jd_short = truncate_to_tokens(jd_text, max_tokens=250)

        pairs = []
        for _, row in df.iterrows():
            pairs.append((jd_short, build_candidate_text(row)))

        ce_scores = ce_model.predict(pairs, batch_size=32, show_progress_bar=False)
        ce_scores = np.array(ce_scores, dtype=np.float32)
        status_lines.append(f"Cross-encoded {len(pairs)} pairs")
    except Exception as e:
        ce_scores = np.zeros(len(df), dtype=np.float32)
        status_lines.append(f"Cross-encoder failed: {e}, using fallback")

    # Signal scores
    signal_scores = np.dot(signal_mat, jd_spec.signal_weights)

    # Composite
    composite = 0.65 * min_max_norm(ce_scores) + 0.35 * signal_scores

    # Sort and build output
    order = np.argsort(-composite)
    top_n = min(len(df), 100)

    output_rows = []
    for rank, idx in enumerate(order[:top_n], start=1):
        row = df.iloc[idx]
        reasoning = fallback_reasoning(row, rank, jd_spec)
        output_rows.append({
            "candidate_id": row["candidate_id"],
            "rank": rank,
            "score": round(float(composite[idx]), 6),
            "reasoning": reasoning,
        })

    result_df = pd.DataFrame(output_rows)

    # Enforce monotonicity
    scores_arr = result_df["score"].values.copy()
    scores_arr = enforce_monotonicity(scores_arr)
    result_df["score"] = np.round(scores_arr, 6)

    elapsed = time.time() - t_start
    status_lines.append(
        f"Runtime: {elapsed:.1f}s | Models: all-MiniLM-L6-v2 + ms-marco-MiniLM-L-6-v2 | No GPU"
    )

    return result_df, "\n".join(status_lines)


def create_app():
    """Create Gradio interface."""
    import gradio as gr

    with gr.Blocks(
        title="Redrob Candidate Ranker",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
        # 🎯 Redrob Candidate Ranking System
        **Intelligent Candidate Discovery & Ranking**

        Upload a JSON file of candidates (max 100) and optionally paste a job description.
        The system will rank candidates using a hybrid pipeline:
        cross-encoder scoring + behavioral signals.

        *CPU-only | No GPU | No external API calls*
        """)

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload candidates JSON (max 100 candidates)",
                    file_types=[".json", ".jsonl"],
                )
                jd_input = gr.Textbox(
                    label="Job Description (leave blank for default Senior AI Engineer JD)",
                    lines=10,
                    placeholder="Paste job description here...",
                )
                rank_btn = gr.Button("🚀 Rank Candidates", variant="primary")

            with gr.Column(scale=2):
                status_output = gr.Textbox(label="Status", lines=5)
                result_table = gr.Dataframe(
                    label="Ranked Candidates (top 10 shown)",
                    headers=["candidate_id", "rank", "score", "reasoning"],
                    wrap=True,
                )

        def process(file, jd):
            result_df, status = rank_candidates(file, jd)
            if result_df.empty:
                return status, pd.DataFrame()
            return status, result_df.head(10)

        rank_btn.click(
            fn=process,
            inputs=[file_input, jd_input],
            outputs=[status_output, result_table],
        )

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=False)
