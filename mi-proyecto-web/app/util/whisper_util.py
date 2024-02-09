import os
from pytube import YouTube
from transformers import pipeline
# Configuración de la pipeline de transcripción
transcription_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-medium")

def download_audio_from_youtube(url, output_path='uploads'):
    # Verifica y crea el directorio 'uploads' si no existe
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    # Asegura el nombre del archivo descargado
    filename = "temp_audio.mp4"
    audio_stream.download(output_path=output_path, filename=filename)
    # Devuelve la ruta completa al archivo descargado
    return os.path.join(output_path, filename)



def transcribe_audio(audio_path):
    result = transcription_pipeline(audio_path)
    os.remove(audio_path)  # Cleanup the downloaded audio file
    return result['text']

def process_youtube_video(url):
    audio_path = download_audio_from_youtube(url, 'uploads')
    transcription = transcribe_audio(audio_path)
    return transcription


