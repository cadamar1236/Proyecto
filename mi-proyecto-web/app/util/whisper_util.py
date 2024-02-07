from pytube import YouTube
import os
import whisper

def download_audio_from_youtube(url, output_path):
    yt = YouTube(url)
    audio_stream = yt.streams.get_audio_only()
    audio_stream.download(output_path=output_path, filename='temp_audio')
    return os.path.join(output_path, 'temp_audio.mp4')

def transcribe_audio(audio_path):
    model = whisper.load_model("base")  # O elige otro modelo según tus recursos
    result = model.transcribe(audio_path)
    os.remove(audio_path)  # Cleanup the downloaded audio file
    return result["text"]

def process_youtube_video(url):
    audio_path = download_audio_from_youtube(url, 'uploads')
    transcription = transcribe_audio(audio_path)
    return transcription

