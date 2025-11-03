from keybert import KeyBERT
import yake
import langid
import re
import spacy
from spacy.matcher import PhraseMatcher
from skillNer.skill_extractor_class import SkillExtractor
from skillNer.general_params import SKILL_DB  # dict in v1.x
# 1️⃣ Example texts
cv_text = """
Erfahrener Softwareentwickler mit Kenntnissen in Python, Django und REST API Entwicklung.
Erfahrung mit Docker, AWS und CI/CD-Pipelines. Fließend in Deutsch und Englisch.
"""

jd_text = """
Wir suchen einen Backend-Entwickler mit Erfahrung in Python, RESTful APIs und AWS Cloud-Diensten.
Kenntnisse in CI/CD und Containerisierung (Docker, Kubernetes) sind von Vorteil.
"""

# 2️⃣ Keyword extraction using YAKE (multilingual)
def extract_yake(text, lang="en", top_k=10):
    kw_extractor = yake.KeywordExtractor(lan=lang, n=1, top=top_k)
    keywords = [kw for kw, score in kw_extractor.extract_keywords(text)]
    return keywords

# 3️⃣ Keyword extraction using KeyBERT (semantic)
def extract_keybert(text, model_name="paraphrase-multilingual-MiniLM-L12-v2", top_k=10):
    kw_model = KeyBERT(model_name)
    keywords = [kw for kw, score in kw_model.extract_keywords(text, top_n=top_k)]
    return keywords

# 4️⃣ Optional language detection
def detect_lang(text):
    lang, _ = langid.classify(text)
    return lang if lang in ["en", "de"] else "en"

# 5️⃣ Master extraction with language switch
def extract_keywords(text, lang=None, top_k=50):
    """
    Extract hard-skill keywords using SkillNER (v1.0.3) with robust fallbacks.
    - Uses requested/detected language ('en'/'de').
    - If SkillNER returns no results, falls back to a direct PhraseMatcher over SKILL_DB.
    - If lang='de' still yields nothing, retries with English model.
    Returns a deduped, lowercased list (optionally trimmed to top_k).
    """
    if not text or not text.strip():
        return []

    def _run_with_lang(model_lang: str):
        # use german language model incase its needed
        model_name = "en_core_web_lg"
        nlp = spacy.load(model_name)
        nlp.max_length = max(nlp.max_length, 2_000_000)

        # 1) Try SkillNER
        extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
        skills = []
        try:
            ann = extractor.annotate(text)
            results = ann.get("results", {})
            print("result", results)
            for group in ("full_matches", "ngram_scored"):
                if group in results:
                    for item in results[group]:
                        val = (item.get("doc_node_value") or "").strip().lower()
                        if val and val not in skills:
                            skills.append(val)
                            print(val)
        except Exception:
            skills = []

        # 2) Fallback: direct PhraseMatcher over SKILL_DB if still empty
        if not skills:
            print("No skills found, using fallback phrasematcher")
            pm = PhraseMatcher(nlp.vocab, attr="LOWER")
            # Build patterns from skill names (filter extremes to keep it snappy)
            names = [name for name in SKILL_DB.keys() if 1 < len(name) < 60]
            # Avoid exceeding add() limits by chunking if needed
            chunk = 2000
            for i in range(0, len(names), chunk):
                docs = [nlp.make_doc(n) for n in names[i:i+chunk]]
                pm.add(f"SKILL_{i//chunk}", docs)

            doc = nlp(text)
            for _, start, end in pm(doc):
                val = doc[start:end].text.strip().lower()
                if val and val not in skills:
                    skills.append(val)
        return skills

    # Resolve language
    code = (lang or detect_lang(text) or "en").lower()

    skills = _run_with_lang(code)
    # If German path yields nothing, try English as a pragmatic fallback
    if not skills and code.startswith("de"):
        skills = _run_with_lang("en")

    # Trim to top_k deterministically
    if top_k and len(skills) > top_k:
        skills = skills[:top_k]

    return skills





# 6️⃣ Run extraction manually with language switch
# cv_keywords = extract_keywords(cv_text, lang="de")  # Force German
# jd_keywords = extract_keywords(jd_text, lang="de")
#
# # 7️⃣ Print results
# print("\n=== CV Keywords (German) ===")
# print(cv_keywords)
#
# print("\n=== JD Keywords (German) ===")
# print(jd_keywords)