"""
precompute.py
Master offline script: runs all Stage 0 steps in order.
No time constraint. Run ONCE before ranking.

Usage:
  python precompute.py --candidates data/candidates.jsonl

Steps:
  1. Parse + enrich candidates -> artifacts/candidates_df.parquet
  2. Detect honeypots -> artifacts/honeypot_flags.npy
  3. Build embeddings -> artifacts/embeddings.npy + artifacts/candidate_ids.npy
  4. Build FAISS index -> artifacts/faiss_index.bin
  5. Build BM25 index -> artifacts/bm25_index.pkl + artifacts/bm25_corpus.pkl
  6. Build signal matrix -> artifacts/signal_matrix.npy + artifacts/signal_names.npy
"""

import argparse
import hashlib
import time
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import set_all_seeds, ensure_artifacts_dir, format_time, ARTIFACTS_DIR


def md5_file(path: Path) -> str:
    """Compute MD5 checksum of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Pre-compute all artifacts for ranking")
    parser.add_argument("--candidates", type=str, required=True,
                        help="Path to candidates.jsonl or candidates.jsonl.gz")
    args = parser.parse_args()

    set_all_seeds(42)
    ensure_artifacts_dir()
    t_total = time.time()

    # ─────────────────────────────────────
    # Step 1: Parse + enrich candidates
    # ─────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 1/6: Parse and enrich candidates")
    print("="*60)
    t0 = time.time()

    from src.preprocess import preprocess_candidates, save_parquet
    df = preprocess_candidates(args.candidates)
    save_parquet(df)

    print(f"  Step 1 completed in {format_time(time.time() - t0)}")

    # ─────────────────────────────────────
    # Step 2: Detect honeypots
    # ─────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 2/6: Detect honeypots")
    print("="*60)
    t0 = time.time()

    from src.honeypot import detect_honeypots, save_honeypot_flags
    flags = detect_honeypots(df)
    save_honeypot_flags(flags)

    print(f"  Step 2 completed in {format_time(time.time() - t0)}")

    # ─────────────────────────────────────
    # Step 3: Build embeddings
    # ─────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 3/6: Build bi-encoder embeddings")
    print("="*60)
    t0 = time.time()

    from src.embed import build_embeddings, save_embeddings, save_faiss_index, build_faiss_index
    from sentence_transformers import SentenceTransformer
    from src.embed import MODEL_NAME

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    embeddings = build_embeddings(df, model)

    candidate_ids = df["candidate_id"].values.astype(str)
    save_embeddings(embeddings, candidate_ids)

    print(f"  Step 3 completed in {format_time(time.time() - t0)}")

    # ─────────────────────────────────────
    # Step 4: Build FAISS index
    # ─────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 4/6: Build FAISS index")
    print("="*60)
    t0 = time.time()

    index = build_faiss_index(embeddings)
    save_faiss_index(index)

    print(f"  Step 4 completed in {format_time(time.time() - t0)}")

    # ─────────────────────────────────────
    # Step 5: Build BM25 index
    # ─────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 5/6: Build BM25 index")
    print("="*60)
    t0 = time.time()

    from src.bm25_index import build_bm25, save_bm25
    bm25, corpus = build_bm25(df)
    save_bm25(bm25, corpus)

    print(f"  Step 5 completed in {format_time(time.time() - t0)}")

    # ─────────────────────────────────────
    # Step 6: Build signal matrix
    # ─────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 6/6: Build signal matrix")
    print("="*60)
    t0 = time.time()

    from src.signal_extractor import build_signal_matrix, save_signal_matrix
    matrix = build_signal_matrix(df)
    save_signal_matrix(matrix)

    print(f"  Step 6 completed in {format_time(time.time() - t0)}")

    # ─────────────────────────────────────
    # Summary
    # ─────────────────────────────────────
    total_time = time.time() - t_total
    print("\n" + "="*60)
    print("  PRE-COMPUTATION COMPLETE")
    print("="*60)
    print(f"  Total runtime: {format_time(total_time)}")
    print(f"\n  Artifacts in {ARTIFACTS_DIR}:")

    for artifact in sorted(ARTIFACTS_DIR.iterdir()):
        size = artifact.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / (1024*1024):.1f} MB"
        else:
            size_str = f"{size / 1024:.1f} KB"
        checksum = md5_file(artifact)
        print(f"    {artifact.name:30s} {size_str:>10s}  md5:{checksum[:12]}")

    print(f"\n  Next step: python rank.py --candidates {args.candidates} "
          f"--jd data/job_description.docx --out submission.csv")


if __name__ == "__main__":
    main()
