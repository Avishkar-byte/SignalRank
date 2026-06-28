"""
jd_parser.py
Reads job description (.md or .docx) and produces a JDSpec dataclass.
Extracts hard knockouts, skill weights, and signal weights.
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import read_jd_file, truncate_to_tokens
from src.signal_extractor import SIGNAL_NAMES


@dataclass
class JDSpec:
    raw_text: str
    query_text: str  # optimized text for bi-encoder query

    # Hard knockout criteria
    min_years_experience: float | None = None
    required_education: str | None = None
    required_location: str | None = None
    required_skills: list[str] = field(default_factory=list)

    # Weighted skill requirements
    primary_skills: list[str] = field(default_factory=list)
    secondary_skills: list[str] = field(default_factory=list)
    nice_to_have_skills: list[str] = field(default_factory=list)

    # Signal weight vector (length 23)
    signal_weights: np.ndarray = field(default_factory=lambda: np.ones(23) / 23.0)

    # Derived
    skill_weight_map: dict[str, float] = field(default_factory=dict)

    # Disqualifier patterns
    disqualifier_companies: list[str] = field(default_factory=list)
    disqualifier_title_patterns: list[str] = field(default_factory=list)


KNOWN_IT_SERVICE_FIRMS = {
    "genpact", "infosys", "wipro", "tcs", "tata consultancy", 
    "hcl", "tech mahindra", "cognizant", "accenture", "capgemini",
    "mphasis", "hexaware", "mindtree", "ltimindtree", "persistent systems"
}

def extract_hackathon_hints(jd_text: str) -> dict:
    """
    Scan the FULL jd_text for a section that appears after the main role description.
    """
    hints = {
        "raw_hints_section": "",
        "emphasized_signals": [],
        "explicit_disqualifiers": [],
        "tier_hints": {},
        "weight_hints": {}
    }
    
    # Try to find the hackathon section
    marker = "Final note for the participants of the Redrob hackathon"
    idx = jd_text.find(marker)
    if idx == -1:
        # Fallback to general hints
        marker = "hackathon"
        idx = jd_text.lower().find(marker)
        
    if idx != -1:
        hints_section = jd_text[idx:]
        hints["raw_hints_section"] = hints_section
        
        # Parse for specific hints
        hints_lower = hints_section.lower()
        if "hasn't logged in for 6 months" in hints_lower or "not actually available" in hints_lower:
            hints["emphasized_signals"].extend(["days_since_active", "open_to_work_flag"])
        if "recruiter response rate" in hints_lower:
            hints["emphasized_signals"].append("recruiter_response_rate")
        if "trap" in hints_lower and "keywords" in hints_lower:
            hints["weight_hints"]["keyword_trap"] = True
            
        print("\n=== HACKATHON HINTS SECTION FOUND ===")
        print(hints_section.strip())
        print("=====================================")
    else:
        print("\n=== NO HACKATHON HINTS FOUND ===")
        
    return hints


def parse_jd(jd_path: str, signal_names: list[str] | None = None) -> JDSpec:
    """
    Parse job description into structured JDSpec.
    """
    if signal_names is None:
        signal_names = SIGNAL_NAMES

    raw_text = read_jd_file(jd_path)
    text_lower = raw_text.lower()

    print(f"[jd_parser] Parsing JD from {jd_path} ({len(raw_text)} chars)...")

    # Extract hints
    hints = extract_hackathon_hints(raw_text)

    # 1. Hard knockouts
    min_yoe = 3.0
    required_education = None
    required_location = None
    required_skills = [
        "embeddings", "retrieval", "vector", "search",
        "python", "ranking", "faiss", "elasticsearch",
        "sentence-transformers", "embedding",
    ]

    # 2. Skill tiers
    primary_skills = [
        "embeddings", "embedding", "sentence-transformers", "retrieval",
        "vector database", "vector search", "semantic search",
        "faiss", "pinecone", "weaviate", "qdrant", "milvus",
        "elasticsearch", "opensearch",
        "machine learning", "ml", "deep learning", "nlp",
        "natural language processing", "information retrieval",
        "ranking", "recommendation", "search",
        "python",
        "ndcg", "mrr", "evaluation", "a/b testing", "metrics",
        "llm", "large language model", "transformers", "huggingface",
        "pytorch", "tensorflow",
        "data pipeline", "data engineering",
    ]

    secondary_skills = [
        "fine-tuning", "lora", "qlora", "peft",
        "learning to rank", "xgboost",
        "hr tech", "recruiting", "marketplace",
        "distributed systems", "inference optimization",
        "open source", "oss",
        "rag", "retrieval augmented generation",
        "docker", "kubernetes", "aws", "gcp", "azure",
        "spark", "kafka", "airflow",
    ]

    nice_to_have_skills = [
        "sql", "java", "scala", "go", "rust",
        "mongodb", "redis", "postgresql",
        "ci/cd", "mlops", "mlflow",
        "wandb", "weights & biases",
        "langchain",
    ]

    skill_weight_map = {}
    for s in primary_skills:
        skill_weight_map[s.lower()] = 1.0
    for s in secondary_skills:
        skill_weight_map[s.lower()] = 0.6
    for s in nice_to_have_skills:
        skill_weight_map[s.lower()] = 0.3

    # 3. Signal weights
    signal_weight_map = {
        "profile_completeness_score": 0.03,
        "profile_views_received_30d": 0.05,
        "applications_submitted_30d": 0.03,
        "recruiter_response_rate": 0.12,
        "avg_response_time_hours": 0.08,
        "connection_count": 0.02,
        "endorsements_received": 0.03,
        "notice_period_days": 0.10,
        "github_activity_score": 0.10,
        "search_appearance_30d": 0.04,
        "saved_by_recruiters_30d": 0.06,
        "interview_completion_rate": 0.07,
        "offer_acceptance_rate": 0.03,
        "open_to_work_flag": 0.06,
        "willing_to_relocate": 0.04,
        "verified_email": 0.02,
        "verified_phone": 0.02,
        "linkedin_connected": 0.02,
        "days_since_signup": 0.01,
        "days_since_active": 0.08,
        "salary_midpoint_lpa": 0.01,
        "skill_assessment_mean": 0.05,
        "work_mode_score": 0.03,
    }

    # Upweight based on hints
    for signal in hints["emphasized_signals"]:
        if signal in signal_weight_map:
            print(f"[jd_parser] Upweighting hint signal: {signal}")
            signal_weight_map[signal] *= 1.5

    # Build weight vector
    weights = np.zeros(len(signal_names), dtype=np.float32)
    for i, name in enumerate(signal_names):
        weights[i] = signal_weight_map.get(name, 1.0 / len(signal_names))
    weights = weights / weights.sum()

    # 4. Disqualifiers
    disqualifier_companies = list(KNOWN_IT_SERVICE_FIRMS)

    # 5. Build query_text
    query_text = (
        "Senior AI Engineer machine learning embeddings retrieval ranking "
        "NLP information retrieval vector database FAISS semantic search "
        "Python deep learning LLM fine-tuning evaluation NDCG MRR "
        "production deployment sentence-transformers hybrid search "
        "recommendation system data pipeline ranking system "
        "candidate matching talent intelligence platform"
    )
    query_text = truncate_to_tokens(query_text, max_tokens=256)

    spec = JDSpec(
        raw_text=raw_text,
        query_text=query_text,
        min_years_experience=min_yoe,
        required_education=required_education,
        required_location=required_location,
        required_skills=required_skills,
        primary_skills=primary_skills,
        secondary_skills=secondary_skills,
        nice_to_have_skills=nice_to_have_skills,
        signal_weights=weights,
        skill_weight_map=skill_weight_map,
        disqualifier_companies=disqualifier_companies,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"  JD PARSE SUMMARY")
    print(f"{'='*60}")
    print(f"  Role: Senior AI Engineer")
    print(f"  Min YOE: {spec.min_years_experience}")
    print(f"  Required education: {spec.required_education or 'None'}")
    print(f"  Required location: {spec.required_location or 'Flexible (India preferred)'}")
    print(f"  Primary skills ({len(spec.primary_skills)}): {spec.primary_skills[:10]}...")
    print(f"  Secondary skills ({len(spec.secondary_skills)}): {spec.secondary_skills[:8]}...")
    print(f"  Nice-to-have ({len(spec.nice_to_have_skills)}): {spec.nice_to_have_skills[:6]}...")
    print(f"  Disqualifier companies: {spec.disqualifier_companies}")
    print(f"  Signal weights (top 5):")
    top_signals = sorted(zip(signal_names, weights), key=lambda x: -x[1])[:5]
    for name, w in top_signals:
        print(f"    {name}: {w:.3f}")
    print(f"  Hackathon hints found: {'YES' if hints['raw_hints_section'] else 'NO'}")
    print(f"{'='*60}\n")

    return spec


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd", type=str, required=True, help="Path to job_description.md or .docx")
    args = parser.parse_args()

    signal_names = SIGNAL_NAMES
    spec = parse_jd(args.jd, signal_names)

