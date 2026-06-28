"""
bm25_index.py
Uses rank_bm25.BM25Okapi.
Tokenizes profile_text by splitting on whitespace and lowercasing.
Pickles the BM25 object and token corpus.
"""

import pickle
import sys
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import ensure_artifacts_dir, tokenize_text, ARTIFACTS_DIR


def build_bm25(df: pd.DataFrame) -> tuple[BM25Okapi, list[list[str]]]:
    """
    Tokenize df['profile_text'], build BM25Okapi.
    Returns (bm25_index, corpus).
    """
    print(f"[bm25] Tokenizing {len(df)} documents...")
    corpus = []
    for text in tqdm(df["profile_text"].fillna(""), desc="Tokenizing"):
        corpus.append(tokenize_text(text))

    print(f"[bm25] Building BM25 index...")
    bm25 = BM25Okapi(corpus)
    print(f"[bm25] BM25 index built: {len(corpus)} documents, avg doc len={bm25.avgdl:.1f}")
    return bm25, corpus


def save_bm25(bm25: BM25Okapi, corpus: list[list[str]]) -> None:
    """Pickle BM25 index and corpus."""
    ensure_artifacts_dir()

    bm25_path = ARTIFACTS_DIR / "bm25_index.pkl"
    corpus_path = ARTIFACTS_DIR / "bm25_corpus.pkl"

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    with open(corpus_path, "wb") as f:
        pickle.dump(corpus, f)

    bm25_mb = bm25_path.stat().st_size / (1024 * 1024)
    corpus_mb = corpus_path.stat().st_size / (1024 * 1024)
    print(f"[bm25] Saved {bm25_path} ({bm25_mb:.1f} MB)")
    print(f"[bm25] Saved {corpus_path} ({corpus_mb:.1f} MB)")


def load_bm25() -> tuple[BM25Okapi, list[list[str]]]:
    """Load pickled BM25 index and corpus."""
    bm25_path = ARTIFACTS_DIR / "bm25_index.pkl"
    corpus_path = ARTIFACTS_DIR / "bm25_corpus.pkl"

    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    with open(corpus_path, "rb") as f:
        corpus = pickle.load(f)

    return bm25, corpus


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=str(ARTIFACTS_DIR / "candidates_df.parquet"))
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    bm25, corpus = build_bm25(df)
    save_bm25(bm25, corpus)
    print("[bm25] Done!")
