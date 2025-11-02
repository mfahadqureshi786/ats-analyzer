import requests

def generate_feedback(cv_text: str) -> str:
    prompt = (
        "You are an experienced HR recruiter. "
        "Read the following CV text and give bullet-point suggestions "
        "to improve clarity, formatting, and keyword usage for ATS optimization.\n\n"
        f"CV:\n{cv_text[:4000]}"  # truncate long CVs
    )

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt}
        )
        # Use .text instead of .json()
        return response.text.strip()
    except Exception as e:
        return f"Error connecting to Mistral: {e}"
