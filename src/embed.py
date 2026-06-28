"""
embed.py
Uses sentence-transformers all-MiniLM-L6-v2 (CPU).
Generates bi-encoder embeddings for all candidates and builds FAISS index.

Outputs:
  artifacts/embeddings.npy     shape (N, 384) float32
  artifacts/candidate_ids.npy  shape (N,) string array
  artifacts/faiss_index.bin    IVF-Flat index
"""

import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import ensure_artifacts_dir, set_all_seeds, ARTIFACTS_DIR


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 512
NLIST = 200  # FAISS IVF clusters (sqrt(100K) ≈ 316, use 200 for speed)
EMBEDDING_DIM = 384


def build_embeddings(df: pd.DataFrame, model: SentenceTransformer | None = None) -> np.ndarray:
    """
    Encode df['profile_text'] in batches.
    Return float32 numpy array shape (len(df), 384).
    Normalize vectors (L2) for cosine similarity via inner product in FAISS.
    """
    set_all_seeds(42)

    if model is None:
        print(f"[embed] Loading model {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME, device="cpu")

    texts = df["profile_text"].fillna("").tolist()
    print(f"[embed] Encoding {len(texts)} texts with batch_size={BATCH_SIZE}...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
        convert_to_numpy=True,
    )

    embeddings = embeddings.astype(np.float32)
    print(f"[embed] Embeddings shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS IndexIVFFlat with inner product metric (cosine on normalized vectors).
    Train on a random sample of 50K vectors.
    """
    set_all_seeds(42)
    faiss.omp_set_num_threads(1)  # Determinism

    n, d = embeddings.shape
    print(f"[embed] Building FAISS IVF-Flat index (n={n}, d={d}, nlist={NLIST})...")

    # Quantizer
    quantizer = faiss.IndexFlatIP(d)  # Inner product (cosine on L2-normalized)
    index = faiss.IndexIVFFlat(quantizer, d, NLIST, faiss.METRIC_INNER_PRODUCT)

    # Train on subset
    train_size = min(50000, n)
    rng = np.random.RandomState(42)
    train_indices = rng.choice(n, size=train_size, replace=False)
    train_vectors = embeddings[train_indices]

    print(f"[embed] Training index on {train_size} vectors...")
    index.train(train_vectors)

    # Add all vectors
    print(f"[embed] Adding {n} vectors to index...")
    index.add(embeddings)

    # Set nprobe for query time
    index.nprobe = 32

    print(f"[embed] Index built: {index.ntotal} vectors, nlist={NLIST}, nprobe={index.nprobe}")

    # Verify: query first embedding, top-1 should be itself
    D, I = index.search(embeddings[:1], 1)
    assert I[0][0] == 0, f"Self-query verification failed: expected index 0, got {I[0][0]}"
    print(f"[embed] Self-query verification passed (score={D[0][0]:.4f})")

    return index


def save_embeddings(embeddings: np.ndarray, candidate_ids: np.ndarray) -> None:
    """Save embeddings and candidate IDs to disk."""
    ensure_artifacts_dir()

    emb_path = ARTIFACTS_DIR / "embeddings.npy"
    ids_path = ARTIFACTS_DIR / "candidate_ids.npy"

    np.save(emb_path, embeddings)
    np.save(ids_path, candidate_ids)

    emb_mb = emb_path.stat().st_size / (1024 * 1024)
    print(f"[embed] Saved {emb_path} ({emb_mb:.1f} MB)")
    print(f"[embed] Saved {ids_path}")


def save_faiss_index(index: faiss.Index) -> None:
    """Save FAISS index to disk."""
    ensure_artifacts_dir()
    index_path = ARTIFACTS_DIR / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    idx_mb = index_path.stat().st_size / (1024 * 1024)
    print(f"[embed] Saved {index_path} ({idx_mb:.1f} MB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=str(ARTIFACTS_DIR / "candidates_df.parquet"))
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    embeddings = build_embeddings(df, model)
    index = build_faiss_index(embeddings)

    candidate_ids = df["candidate_id"].values.astype(str)
    save_embeddings(embeddings, candidate_ids)
    save_faiss_index(index)
    print("[embed] Done!")
