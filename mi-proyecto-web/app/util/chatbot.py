import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_mistral_response(prompt):
    YOUR_HUGGING_FACE_API_TOKEN = os.getenv("YOUR_HUGGING_FACE_API_TOKEN")
    
    # Verificar que el token de API está presente
    if not YOUR_HUGGING_FACE_API_TOKEN:
        return "El token de API no está definido en las variables de entorno."
    
    API_URL = "https://api-inference.huggingface.co/models/mistralai/mistral-7B-instruct-v0.2"
    headers = {"Authorization": f"Bearer {YOUR_HUGGING_FACE_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_length": 50, "temperature": 0.7},
        "options": {"use_cache": False},
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        # Verificar que la respuesta es exitosa
        if response.status_code == 200:
            result = response.json()
            return result[0]['generated_text']
        else:
            # Manejar respuestas que no sean código 200
            return f"Error {response.status_code}: {response.reason}"
    
    except requests.exceptions.RequestException as e:
        # Manejar errores de solicitud, como problemas de conectividad
        return f"Error de solicitud HTTP: {e}"
