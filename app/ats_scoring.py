from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT

# Load models once (lightweight)
model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=model)

# Example keywords (you can later expand or load from job descriptions)
TARGET_KEYWORDS = [
    "python", "fastapi", "machine learning", "sql",
    "teamwork", "data analysis", "communication", "docker", "git"
]


def calculate_ats_score(text: str, jd_target_keywords) -> dict:
    """Calculate a simple ATS score based on keywords + semantic similarity.
    Handles empty JD keyword lists safely.
    """
    # Guard clause: no JD keywords
    if not jd_target_keywords:
        return {
            "keyword_score": 0.0,
            "semantic_score": 0.0,
            "final_score": 0.0,
            "matched_keywords": []
        }

    text_lower = text.lower()

    # Case-insensitive exact keyword matches
    matched_keywords = [kw for kw in jd_target_keywords if kw.lower() in text_lower]

    # Keyword coverage score
    keyword_score = (len(matched_keywords) / len(jd_target_keywords)) * 100

    # Semantic similarity (resume text vs JD skill list)
    try:
        doc_emb = model.encode(text, convert_to_tensor=True)
        job_emb = model.encode(" ".join(jd_target_keywords), convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(doc_emb, job_emb).item() * 100
    except Exception:
        similarity = 0.0

    # Weighted final score (60% keywords + 40% semantic)
    final_score = round((keyword_score * 0.6 + similarity * 0.4), 2)

    return {
        "keyword_score": round(keyword_score, 2),
        "semantic_score": round(similarity, 2),
        "final_score": final_score,
        "matched_keywords": matched_keywords
    }
