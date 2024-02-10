from transformers import pipeline

def load_mistral_model():
    # Cargar el modelo de Mistral AI para generación de texto
    chatbot = pipeline("text-generation", model="mistralai/mistral-7B-instruct-v0.2")
    return chatbot

chatbot = load_mistral_model()

def get_mistral_response(prompt, max_length=50, temperature=0.7, num_return_sequences=1):
    try:
        # Generar respuesta(s) con control sobre la diversidad y evitando repeticiones
        responses = chatbot(prompt, max_length=max_length, temperature=temperature,
                            num_return_sequences=num_return_sequences, clean_up_tokenization_spaces=True,
                            no_repeat_ngram_size=2)
        
        # Si generamos varias respuestas, elegimos la primera como nuestra respuesta.
        return responses[0]['generated_text']
    except Exception as e:
        print(f"Error al generar respuesta: {e}")
        # Devolver un mensaje de error o fallback
        return "Lo siento, no puedo responder a eso en este momento."


