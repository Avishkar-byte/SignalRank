# SignalRank

Hybrid AI pipeline for intelligent candidate discovery and ranking. Built for the Redrob Hackathon - Intelligent Candidate Discovery and Ranking Challenge.

> Built for **INDIA RUNS**

---

## What it does

SignalRank takes a pool of 100,000 candidate profiles and a job description, and produces a ranked shortlist of the top 100 candidates in under 150 seconds on a CPU - with no GPU, no external API calls, and no cloud dependency during ranking.

The system goes beyond keyword matching by combining semantic search, lexical retrieval, cross-encoder re-ranking, and a 23-signal behavioral layer that captures candidate availability, engagement, and career trajectory.

---

## Architecture

### Stage 0 (Offline, unlimited time)
- Parse + enrich 100K candidates
- Detect honeypot profiles (rule-based, 6 hard + 6 soft rules)
- Bi-encoder embeddings (all-MiniLM-L6-v2) + FAISS IVF-Flat index
- BM25 index over profile text
- 23-signal behavioral matrix

### Stage 1 (Online, <10s)
- Parse JD - hard knockouts, skill tiers, signal weights, recency hints
- FAISS semantic retrieval - top 2000
- BM25 lexical retrieval - top 2000
- Reciprocal Rank Fusion - top 700 pool

### Stage 2 (Online, ~110s)
- Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2) on 700 pairs
- Behavioral signal scoring - dot product against JD-derived weight vector
- Score fusion - CE x 0.55 + signal x 0.30 + BM25 x 0.15
- Honeypot penalty - 0.05-0.95x multiplier based on confidence
- Consulting-firm disqualifier - 0.1x multiplier

### Stage 3 (Online, <1s)
- Sort, enforce monotonicity, join reasoning cache
- Output ranked CSV

---

## Results

| Metric | Value |
|--------|-------|
| Total runtime | ~148s on CPU |
| Candidates processed | 100,000 |
| Honeypots flagged | 215 / 100,000 (0.21%) |
| Honeypots in top 100 | 0 |
| Output rows | 100 (ranks 1-100) |

Top 5 ranked candidates:

| Rank | Candidate | Role | Company | YOE |
|------|-----------|------|---------|-----|
| 1 | CAND_0046064 | Senior NLP Engineer | Salesforce | 8.9 |
| 2 | CAND_0011687 | Senior NLP Engineer | Niramai | 7.8 |
| 3 | CAND_0081846 | Lead AI Engineer | Razorpay | 6.7 |
| 4 | CAND_0002025 | Senior AI Engineer | Apple | 5.9 |
| 5 | CAND_0033861 | Senior NLP Engineer | Mad Street Den | 8.0 |

---

## Setup

```bash
git clone https://github.com/Avishkar-byte/SignalRank.git
cd SignalRank
pip install -r requirements.txt
```

Place the hackathon bundle files in `data/`:
- `data/candidates.jsonl.gz`
- `data/job_description.md`
- `data/redrob_signals_doc.md`
- `data/candidate_schema.json`

---

## Usage

### Step 1 - Pre-compute artifacts (run once, ~20 minutes, no time constraint)

```bash
python precompute.py --candidates data/candidates.jsonl.gz
```

Outputs all indexes and embeddings to `artifacts/`.

### Step 2 - Generate reasoning cache (run once, offline, optional LLM API key)

```bash
python generate_reasoning.py --jd data/job_description.md
```

### Step 3 - Rank candidates (the submission step, under 5 minutes, CPU only, no network)

```bash
python rank.py --candidates data/candidates.jsonl.gz --jd data/job_description.md --out submission.csv
```

### Step 4 - Validate

```bash
python validate_submission.py --submission submission.csv --candidates data/candidates.jsonl.gz
```

---

## Runtime breakdown

| Stage | Step | Time |
|-------|------|------|
| Load | Artifact loading | ~29s |
| Stage 1 | FAISS + BM25 + RRF | ~7.5s |
| Stage 2 | Cross-encoder (700 pairs) | ~110s |
| Stage 2 | Score fusion | ~0.4s |
| Stage 3 | Output + reasoning join | ~0.5s |
| Total | - | ~148s |

---

## Models

| Model | Purpose | Size | GPU required |
|-------|---------|------|-------------|
| all-MiniLM-L6-v2 | Bi-encoder embeddings | 80 MB | No |
| ms-marco-MiniLM-L-6-v2 | Cross-encoder re-ranking | 68 MB | No |

---

## Compute constraints (met)

- Runtime under 300 seconds - PASS (~148s)
- 16 GB RAM - PASS
- CPU only - PASS
- No external API calls during ranking - PASS
- Deterministic output - PASS

---

## Sandbox

Live demo on HuggingFace Spaces (accepts up to 100 candidates):
https://huggingface.co/spaces/aj-2004/SignalRank

---

## Tech stack

- sentence-transformers (bi-encoder + cross-encoder)
- faiss-cpu (ANN indexing)
- rank-bm25 (lexical retrieval)
- pandas, numpy, scikit-learn
- gradio (sandbox demo)

---

## Hackathon

Redrob Intelligent Candidate Discovery and Ranking Challenge
Submission by Avishkar Jaiswal - VIT Chennai
ORCID: 0009-0004-1378-1202
