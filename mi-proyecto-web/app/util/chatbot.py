import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_mistral_response(prompt):
    # Recuperar el token de API desde las variables de entorno
    YOUR_HUGGING_FACE_API_TOKEN = os.getenv("YOUR_HUGGING_FACE_API_TOKEN")
    API_URL = "https://api-inference.huggingface.co/models/mistralai/mistral-7B-instruct-v0.2"
    headers = {"Authorization": f"Bearer {YOUR_HUGGING_FACE_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_length": 50, "temperature": 0.7},
        "options": {"use_cache": False},
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()

    return result[0]['generated_text'] if response.status_code == 200 else "No se pudo generar una respuesta."