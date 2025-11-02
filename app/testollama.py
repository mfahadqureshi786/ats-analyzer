import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "mistral",
    "prompt": "Hello, can you respond?"
}

try:
    response = requests.post(url, json=payload, timeout=10)
    print("Status code:", response.status_code)
    print("Response text:", response.text)
except Exception as e:
    print("Error connecting to Ollama:", e)
