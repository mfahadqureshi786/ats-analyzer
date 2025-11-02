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
    """Calculate a simple ATS score based on keywords + semantic similarity."""
    text_lower = text.lower()

    # Match exact keywords
    matched_keywords = [kw for kw in jd_target_keywords if kw in text_lower]
    keyword_score = len(matched_keywords) / len(jd_target_keywords) * 100

    # Semantic similarity (text vs target skill set)
    doc_emb = model.encode(text, convert_to_tensor=True)
    job_emb = model.encode(" ".join(jd_target_keywords), convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(doc_emb, job_emb).item() * 100

    # Combine scores
    final_score = round((keyword_score * 0.6 + similarity * 0.4), 2)

    return {
        "keyword_score": round(keyword_score, 2),
        "semantic_score": round(similarity, 2),
        "final_score": final_score,
        "matched_keywords": matched_keywords
    }