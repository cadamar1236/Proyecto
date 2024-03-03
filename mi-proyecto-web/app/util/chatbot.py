import requests
import os
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# Carga las variables de entorno
from dotenv import load_dotenv
load_dotenv()

# Asume que ya tienes configurada la autenticación con Hugging Face y que la variable
# de entorno 'HUGGING_FACE_API_TOKEN' contiene tu token de Hugging Face.
token = os.getenv('YOUR_HUGGING_FACE_API_TOKEN')

# Asegúrate de que el nombre del modelo corresponde al modelo que deseas usar.
# Reemplaza 'tu_modelo' y 'tu_tokenizador' con los nombres correctos del modelo y el tokenizador.
llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-2-70b-chat-hf",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 10},
    use_auth_token= token
)

# Crear cadena con el modelo y un template de prompt
template = """Question: {question}  Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)
chat_chain = prompt | llm

def chat_with_llama(pdf_context, question):
    # Generar la respuesta usando la cadena de LangChain
    response = chat_chain.invoke({"question": f"{pdf_context}\n\n{question}"})
    return response.strip().split('\n')[-1]  # Retorna solo la respuesta final


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
