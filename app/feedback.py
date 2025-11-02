import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

def generate_feedback(cv_text: str) -> str:
    """
    Sends CV text to local Mistral (via Ollama) and gets improvement suggestions.
    Returns plain text feedback.
    """
    prompt = (
        "You are an experienced HR recruiter. "
        "Read the following CV text and provide bullet-point suggestions "
        "to improve clarity, formatting, and keyword usage for ATS optimization.\n\n"
        f"CV:\n{cv_text[:4000]}"  # truncate long CVs to fit context window
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt},
            timeout=10
        )

        # Check HTTP status
        if response.status_code != 200:
            return f"Ollama returned status code {response.status_code}"

        # Use .text instead of .json() to avoid JSONDecodeError
        feedback_raw = response.text
        # Clean up extra newlines or spaces
        feedback_clean = "\n".join(line.strip() for line in feedback_raw.splitlines() if line.strip())
        return feedback_clean or "No feedback generated."

    except requests.exceptions.Timeout:
        return "Ollama request timed out. Try again."
    except requests.exceptions.ConnectionError:
        return "Cannot connect to Ollama. Make sure Mistral is running."
    except Exception as e:
        return f"Unexpected error: {e}"