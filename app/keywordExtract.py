from keybert import KeyBERT
import yake
import langid

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
def extract_keywords(text, lang=None, top_k=10):
    """
    Extract keywords from text.
    lang: "en" for English, "de" for German, or None for auto-detect.
    """
    # Auto-detect if not provided
    if not lang:
        lang = detect_lang(text)
    print(f"🗣 Detected/Selected language: {lang}")

    # YAKE language code is usually 'en' or 'de'
    yake_keywords = extract_yake(text, lang=lang, top_k=top_k)
    bert_keywords = extract_keybert(text, top_k=top_k)
    # Normalize to lowercase before deduplication
    combined = [kw.lower().strip() for kw in (yake_keywords + bert_keywords)]
    # Merge and deduplicate
    combined = list(set(combined))
    return combined

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