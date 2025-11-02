from fastapi import FastAPI, UploadFile
import pdfplumber
import docx
from app.ats_scoring import calculate_ats_score
from app.feedback import generate_feedback

app = FastAPI(title="Local ATS Analyzer")

@app.get("/")
def home():
    return {"message": "ATS Analyzer API is running!"}


@app.post("/analyze_cv/")
async def analyze_cv(file: UploadFile):
    """Upload a CV in PDF or DOCX, analyze it for ATS score + feedback."""

    # Extract text
    if file.filename.endswith(".pdf"):
        with pdfplumber.open(file.file) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif file.filename.endswith(".docx"):
        document = docx.Document(file.file)
        text = "\n".join([para.text for para in document.paragraphs])
    else:
        return {"error": "Unsupported file type. Upload a PDF or DOCX."}

    # Run ATS logic
    ats_result = calculate_ats_score(text)

    # Get AI feedback from Mistral (Ollama)
    feedback = generate_feedback(text)

    return {
        "ats_score": ats_result,
        "feedback": feedback
    }
