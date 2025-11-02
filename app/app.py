import streamlit as st
import pandas as pd
import re
from io import BytesIO
from PyPDF2 import PdfReader

# Import your custom functions
from ats_scoring import calculate_ats_score
from feedback import generate_feedback
from keywordExtract import detect_lang, extract_keywords

# ---------------------------------------------------------
# Streamlit Setup
# ---------------------------------------------------------
st.set_page_config(page_title="PDF → ATS Analyzer", page_icon="📄", layout="centered")

st.title("📄 ATS Analyzer (Resume vs Job Description)")
st.caption("Upload a PDF resume and paste a job description — get ATS match, language, and keyword coverage.")

# ---------------------------------------------------------
# Upload + Inputs
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

with col2:
    job_description_input = st.text_area(
        "Job Description:",
        key="job_desc_txt_area",
        value="",
        height=200,
        placeholder="Paste the job description here...",
    )

# Placeholder for ATS Match score
metric_slot = st.empty()

# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------
def extract_text_from_pdf(file_obj) -> str:
    """Read text from a PDF file using PyPDF2"""
    try:
        reader = PdfReader(file_obj)
        texts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            texts.append(page_text)
        return "\n".join(texts).strip()
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        return ""

# def basic_summary(text: str, max_chars: int = 500) -> str:
#     cleaned = re.sub(r"\s+", " ", text).strip()
#     if len(cleaned) <= max_chars:
#         return cleaned
#     return cleaned[:max_chars] + "..."

# ---------------------------------------------------------
# Main logic
# ---------------------------------------------------------
if uploaded is not None:
    # Read PDF
    pdf_bytes = uploaded.read()
    file_like = BytesIO(pdf_bytes)

    cv_text = extract_text_from_pdf(file_like)

    if not cv_text:
        st.warning("No extractable text found. The PDF may be scanned or image-based.")
        metric_slot.metric("ATS Match", "0%", "")
    else:
        # Detect language
        lang = detect_lang(cv_text)
        st.success(f"Detected language: **{lang}**")

        # Extract keywords from job description
        jd_text = job_description_input.strip()
        if not jd_text:
            st.warning("Please enter a job description first.")
        else:
            job_description_keywords = extract_keywords(jd_text, lang)

            # Run ATS score calculation
            ats_result = calculate_ats_score(cv_text, job_description_keywords)
            ats_score = ats_result.get("final_score", 0)
            metric_slot.metric("ATS Match", f"{ats_score}%", "")

            # Optionally get feedback
            # feedback = generate_feedback(cv_text)

            # -------------------------------
            # Display Keyword Coverage
            # -------------------------------
            st.subheader("📊 Keyword Coverage (Hard Skills)")
            matched = set(ats_result.get("matched_keywords", []))
            required = set(job_description_keywords)
            missing = sorted(required - matched)

            coverage_data = []
            for kw in sorted(required):
                coverage_data.append({
                    "Keyword": kw,
                    "Matched": "✅ Yes" if kw in matched else "❌ No"
                })
            df = pd.DataFrame(coverage_data)

            st.dataframe(df, use_container_width=True)

            if missing:
                st.warning(f"{len(missing)} hard skills not found in resume.")
                st.write("**Missing Keywords:**")
                st.write(", ".join(missing))
            else:
                st.success("All job description hard skills are covered!")

            # -------------------------------
            # ATS Result Details
            # -------------------------------
            st.subheader("📄 ATS Result Summary")
            st.text_area("ATS Result:", value=str(ats_result), height=200)

else:
    st.info("Upload a PDF resume to get started.")
    metric_slot.metric("ATS Match", "0%", "")

st.markdown("""
---
**Tips**
- To detect language properly, ensure your resume is text-based (not image-only).
- To improve ATS match, include missing technical skills naturally in your CV.
""")
