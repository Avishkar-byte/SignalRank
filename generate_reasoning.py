"""
generate_reasoning.py
Standalone script: run OFFLINE after artifacts and scored ranking exist.
Generates and caches reasoning for top-200 candidates.

Usage:
  python generate_reasoning.py --jd data/job_description.docx --provider rule_based

This must be run before rank.py to populate artifacts/reasoning_cache.json.
rank.py will use fallback reasoning for any candidate not in the cache.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import set_all_seeds, format_time, tokenize_text, ARTIFACTS_DIR
from src.jd_parser import parse_jd
from src.signal_extractor import SIGNAL_NAMES
from src.retrieve import faiss_retrieve, bm25_retrieve, reciprocal_rank_fusion, apply_hard_knockouts
from src.rerank import cross_encode, CE_MODEL
from src.score_fusion import compute_composite_scores
from src.reasoning import generate_reasoning_batch, save_reasoning_cache
from src.embed import MODEL_NAME


def main():
    parser = argparse.ArgumentParser(description="Generate reasoning for top candidates")
    parser.add_argument("--jd", type=str, required=True, help="Path to job description")
    parser.add_argument("--provider", type=str, default="rule_based",
                        choices=["rule_based", "openai", "anthropic", "gemini"],
                        help="LLM provider for reasoning generation")
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--artifacts", type=str, default=str(ARTIFACTS_DIR))
    args = parser.parse_args()

    set_all_seeds(42)
    artifacts_dir = Path(args.artifacts)
    t_start = time.time()

    print("[reasoning] Loading artifacts...")

    # Load what we need
    df = pd.read_parquet(artifacts_dir / "candidates_df.parquet")
    signal_matrix = np.load(artifacts_dir / "signal_matrix.npy")
    signal_names = np.load(artifacts_dir / "signal_names.npy", allow_pickle=True).tolist()
    honeypot_flags = np.load(artifacts_dir / "honeypot_flags.npy")
    candidate_ids = np.load(artifacts_dir / "candidate_ids.npy", allow_pickle=True)

    faiss_index = faiss.read_index(str(artifacts_dir / "faiss_index.bin"))
    faiss_index.nprobe = 32
    faiss.omp_set_num_threads(1)

    with open(artifacts_dir / "bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)

    # Load models
    st_model = SentenceTransformer(MODEL_NAME, device="cpu")
    ce_model = CrossEncoder(CE_MODEL, max_length=512, device="cpu")

    # Parse JD
    jd_spec = parse_jd(args.jd, signal_names)

    # Run ranking to get scored candidates
    print("[reasoning] Running ranking pipeline...")
    faiss_results = faiss_retrieve(jd_spec, faiss_index, st_model)
    bm25_results = bm25_retrieve(jd_spec, bm25, tokenize_text)
    pool = reciprocal_rank_fusion(faiss_results, bm25_results)
    pool = apply_hard_knockouts(pool, df, jd_spec)

    pool_indices = [idx for idx, _ in pool]
    pool_rrf_scores = np.array([score for _, score in pool], dtype=np.float32)
    pool_df = df.iloc[pool_indices].reset_index(drop=True)

    ce_scores = cross_encode(jd_spec, pool_df, ce_model)

    scored_df = compute_composite_scores(
        pool_indices=pool_indices,
        pool_df=pool_df,
        ce_scores=ce_scores,
        signal_matrix=signal_matrix,
        bm25_rrf_scores=pool_rrf_scores,
        honeypot_flags=honeypot_flags,
        jd_spec=jd_spec,
    )

    # Generate reasoning
    print(f"\n[reasoning] Generating reasoning for top-{args.top_n} candidates...")
    cache = generate_reasoning_batch(
        scored_df=scored_df,
        full_df=df,
        jd_spec=jd_spec,
        top_n=args.top_n,
        provider=args.provider,
    )

    save_reasoning_cache(cache)

    elapsed = time.time() - t_start
    print(f"\n[reasoning] Done in {format_time(elapsed)}")
    print(f"[reasoning] Cache has {len(cache)} entries at {ARTIFACTS_DIR / 'reasoning_cache.json'}")


if __name__ == "__main__":
    main()
