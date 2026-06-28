"""
rank.py
THE SCRIPT THAT RUNS IN UNDER 5 MINUTES ON CPU.
Loads all pre-computed artifacts and produces the submission CSV.

Usage:
  python rank.py --candidates data/candidates.jsonl --jd data/job_description.docx --out submission.csv

Arguments:
  --candidates   path to candidates.jsonl (or .jsonl.gz)
  --jd           path to job_description.md or .docx
  --out          output CSV path (default: submission.csv)
  --top-n        number of candidates to output (default: 100)
  --artifacts    path to artifacts directory (default: ./artifacts)
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import set_all_seeds, format_time, tokenize_text, ARTIFACTS_DIR
from src.jd_parser import parse_jd, JDSpec
from src.signal_extractor import SIGNAL_NAMES
from src.retrieve import faiss_retrieve, bm25_retrieve, reciprocal_rank_fusion, apply_hard_knockouts
from src.rerank import cross_encode, CE_MODEL
from src.score_fusion import compute_composite_scores, enforce_monotonicity
from src.reasoning import load_reasoning_cache, fallback_reasoning
from src.embed import MODEL_NAME


def main():
    parser = argparse.ArgumentParser(description="Rank candidates against a JD")
    parser.add_argument("--candidates", type=str, required=True)
    parser.add_argument("--jd", type=str, required=True)
    parser.add_argument("--out", type=str, default="submission.csv")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--artifacts", type=str, default=str(ARTIFACTS_DIR))
    args = parser.parse_args()

    set_all_seeds(42)
    artifacts_dir = Path(args.artifacts)

    print("="*60)
    print("  REDROB CANDIDATE RANKING PIPELINE")
    print("  CPU-only | No network | <5 min target")
    print("="*60)

    T_START = time.time()

    # ═════════════════════════════════════
    # LOAD ARTIFACTS
    # ═════════════════════════════════════
    t0 = time.time()
    print("\n[rank] Loading artifacts...")

    # Load parquet
    df = pd.read_parquet(artifacts_dir / "candidates_df.parquet")
    print(f"  candidates_df: {len(df)} rows")

    # Load embeddings (only metadata — we load pool subset later for memory safety)
    embeddings = np.load(artifacts_dir / "embeddings.npy", mmap_mode="r")
    print(f"  embeddings: {embeddings.shape}")

    # Load FAISS index
    faiss_index = faiss.read_index(str(artifacts_dir / "faiss_index.bin"))
    faiss_index.nprobe = 32
    faiss.omp_set_num_threads(1)
    print(f"  faiss_index: {faiss_index.ntotal} vectors")

    # Load BM25
    with open(artifacts_dir / "bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    print(f"  bm25_index loaded")

    # Load signal matrix
    signal_matrix = np.load(artifacts_dir / "signal_matrix.npy")
    print(f"  signal_matrix: {signal_matrix.shape}")

    # Load signal names
    signal_names = np.load(artifacts_dir / "signal_names.npy", allow_pickle=True).tolist()

    # Load honeypot penalties
    honeypot_penalties = np.load(artifacts_dir / "honeypot_penalties.npy")
    print(f"  honeypot_penalties loaded, {int((honeypot_penalties < 1.0).sum())} penalized")

    # Load candidate IDs
    candidate_ids = np.load(artifacts_dir / "candidate_ids.npy", allow_pickle=True)

    # Load ML models
    print(f"  Loading SentenceTransformer ({MODEL_NAME})...")
    st_model = SentenceTransformer(MODEL_NAME, device="cpu")

    print(f"  Loading CrossEncoder ({CE_MODEL})...")
    ce_model = CrossEncoder(CE_MODEL, max_length=512, device="cpu")

    t_load = time.time() - t0
    print(f"\n[rank] Artifacts loaded in {format_time(t_load)}")

    # ═════════════════════════════════════
    # PARSE JD
    # ═════════════════════════════════════
    t0 = time.time()
    print("\n[rank] Parsing job description...")

    jd_spec = parse_jd(args.jd, signal_names)

    t_jd = time.time() - t0
    print(f"[rank] JD parsed in {format_time(t_jd)}")

    # ═════════════════════════════════════
    # STAGE 1: RETRIEVE
    # ═════════════════════════════════════
    t0 = time.time()
    print("\n[rank] STAGE 1: Dual retrieval + RRF fusion...")

    faiss_results = faiss_retrieve(jd_spec, faiss_index, st_model)
    bm25_results = bm25_retrieve(jd_spec, bm25, tokenize_text)
    pool = reciprocal_rank_fusion(faiss_results, bm25_results)
    pool = apply_hard_knockouts(pool, df, jd_spec)

    pool_indices = [idx for idx, _ in pool]
    pool_rrf_scores = np.array([score for _, score in pool], dtype=np.float32)

    t_retrieve = time.time() - t0
    print(f"\n[rank] Retrieved {len(pool)} candidates in {format_time(t_retrieve)}")

    # ═════════════════════════════════════
    # STAGE 2a: CROSS-ENCODE
    # ═════════════════════════════════════
    t0 = time.time()
    print("\n[rank] STAGE 2a: Cross-encoder scoring...")

    pool_df = df.iloc[pool_indices].reset_index(drop=True)
    ce_scores = cross_encode(jd_spec, pool_df, ce_model)

    t_ce = time.time() - t0
    print(f"[rank] Cross-encoded {len(pool)} pairs in {format_time(t_ce)}")

    # ═════════════════════════════════════
    # STAGE 2b: SCORE FUSION
    # ═════════════════════════════════════
    t0 = time.time()
    print("\n[rank] STAGE 2b: Score fusion...")

    scored_df = compute_composite_scores(
        pool_indices=pool_indices,
        pool_df=pool_df,
        ce_scores=ce_scores,
        signal_matrix=signal_matrix,
        bm25_rrf_scores=pool_rrf_scores,
        honeypot_penalties=honeypot_penalties,
        jd_spec=jd_spec,
        signal_names=signal_names,
    )

    t_fusion = time.time() - t0
    print(f"[rank] Score fusion complete in {format_time(t_fusion)}")

    # ═════════════════════════════════════
    # STAGE 3: OUTPUT
    # ═════════════════════════════════════
    t0 = time.time()
    print(f"\n[rank] STAGE 3: Building output (top {args.top_n})...")

    top_n = scored_df.head(args.top_n).copy()

    # Load reasoning cache
    reasoning_cache = load_reasoning_cache()
    cached_count = 0

    # Generate reasoning for each candidate
    reasonings = []
    for rank_idx in range(len(top_n)):
        row_scored = top_n.iloc[rank_idx]
        cid = row_scored["candidate_id"]
        rank = rank_idx + 1

        if cid in reasoning_cache:
            reasonings.append(reasoning_cache[cid])
            cached_count += 1
        else:
            # Fallback reasoning
            arr_idx = int(row_scored["array_index"])
            if arr_idx < len(df):
                full_row = df.iloc[arr_idx]
            else:
                full_row = row_scored
            reasoning = fallback_reasoning(full_row, rank, jd_spec, row_scored)
            reasonings.append(reasoning)

    print(f"  Reasoning: {cached_count} cached, {len(reasonings) - cached_count} generated")

    # Build output DataFrame
    scores = top_n["composite_score"].values.copy()

    # Enforce monotonicity
    scores = enforce_monotonicity(scores)

    # Round to 6 decimal places
    scores = np.round(scores, 6)

    # Final monotonicity check after rounding
    for i in range(1, len(scores)):
        if scores[i] >= scores[i-1]:
            scores[i] = scores[i-1] - 0.000001

    output = pd.DataFrame({
        "candidate_id": top_n["candidate_id"].values,
        "rank": list(range(1, len(top_n) + 1)),
        "score": scores,
        "reasoning": reasonings,
    })

    # ── Validation ──
    assert len(output) == args.top_n, f"Expected {args.top_n} rows, got {len(output)}"
    assert set(output["rank"]) == set(range(1, args.top_n + 1)), "Ranks must be 1-N each exactly once"
    for i in range(len(output) - 1):
        assert output.iloc[i]["score"] >= output.iloc[i+1]["score"], \
            f"Score not non-increasing at rank {i+1}: {output.iloc[i]['score']} < {output.iloc[i+1]['score']}"

    # Verify all candidate_ids exist in the dataset
    valid_ids = set(candidate_ids)
    for cid in output["candidate_id"]:
        assert cid in valid_ids, f"candidate_id {cid} not found in candidates data"

    # Top-100 Honeypot Check
    from src.honeypot import detect_honeypots, check_top100_for_honeypots
    # We load detailed honeypot results from the JSON if it exists to pass to check_top100
    hp_json_path = artifacts_dir / "honeypot_details.json"
    if hp_json_path.exists():
        import json
        from src.honeypot import HoneypotResult
        with open(hp_json_path, "r", encoding="utf-8") as f:
            hp_dict = json.load(f)
        hp_results = {}
        for cid, d in hp_dict.items():
            hp_results[cid] = HoneypotResult(
                candidate_id=cid,
                is_flagged=True,
                confidence=d.get("confidence", 1.0),
                triggered_rules=d.get("triggered_rules", []),
                penalty_multiplier=d.get("penalty_multiplier", 1.0)
            )
        check_top100_for_honeypots(output, hp_results)

    # Write CSV
    output.to_csv(args.out, index=False, encoding="utf-8")

    t_output = time.time() - t0
    print(f"[rank] Submission written to {args.out} in {format_time(t_output)}")

    # ═════════════════════════════════════
    # TOTAL TIME SUMMARY
    # ═════════════════════════════════════
    total = time.time() - T_START
    print(f"\n{'='*60}")
    print(f"  RANKING COMPLETE")
    print(f"{'='*60}")
    print(f"  Load artifacts:   {format_time(t_load)}")
    print(f"  JD parsing:       {format_time(t_jd)}")
    print(f"  Retrieval (RRF):  {format_time(t_retrieve)}")
    print(f"  Cross-encoder:    {format_time(t_ce)}")
    print(f"  Score fusion:     {format_time(t_fusion)}")
    print(f"  Output:           {format_time(t_output)}")
    print(f"  -------------------------")
    print(f"  TOTAL:            {format_time(total)}  [limit: 300s]")
    print(f"{'='*60}")

    if total > 270:
        print("  [!] WARNING: Close to 5-minute limit!")
    elif total > 300:
        print("  [x] EXCEEDED 5-minute limit!")
    else:
        print(f"  [OK] Well within budget ({total/300*100:.0f}% of limit)")

    # Print top-10 preview with disqualifier check
    print(f"\n  Top-10 candidates:")
    from src.jd_parser import KNOWN_IT_SERVICE_FIRMS
    for _, row in output.head(10).iterrows():
        cid = row['candidate_id']
        reasoning = row['reasoning']
        score = row['score']
        
        # Simple disqualifier check by looking for the companies in reasoning/data
        # We can extract the company name if it's in the reasoning, or just do a quick string match
        disq_warning = ""
        reasoning_lower = reasoning.lower()
        for c in KNOWN_IT_SERVICE_FIRMS:
            if c in reasoning_lower:
                disq_warning = f" [! WARNING: Possible Disqualifier '{c}']"
                break
                
        print(f"    Rank {row['rank']:3d} | {cid} | "
              f"score={score:.6f}{disq_warning}")
        print(f"      {reasoning[:120]}...")


if __name__ == "__main__":
    main()
