from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from .forms import AudioUploadForm, YouTubeForm, PDFUploadForm  # Asegúrate de incluir PDFUploadForm
from werkzeug.utils import secure_filename  # Importación para manejar nombres de archivo seguros
from .util import whisper_util
import os
from app import app
from transformers import pipeline
from datetime import datetime


@app.route('/')
def index():
    return render_template('base.html', now=datetime.now())

@app.route('/upload', methods=['GET', 'POST'])
def upload_audio():
    form = AudioUploadForm()
    if form.validate_on_submit():
        audio_file = form.audio.data
        audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio_file.filename))
        audio_file.save(audio_file_path)
        
        # Transcribir el audio
        transcription = whisper_util.transcribe_audio(audio_file_path)
        
        # Guardar la transcripción en un archivo
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        transcription_filename = f"transcription_{timestamp}.txt"
        transcription_filepath = os.path.join('transcriptions', transcription_filename)
        with open(transcription_filepath, 'w') as transcription_file:
            transcription_file.write(transcription)
        
        # Cambiar la respuesta para incluir también el nombre del archivo de transcripción
        return render_template('transcription.html', transcription=transcription, transcription_filename=transcription_filename)
    
    return render_template('upload.html', form=form)

@app.route('/transcribe_youtube', methods=['GET', 'POST'])
def transcribe_youtube():
    form = YouTubeForm()  # Asumimos que ya tienes definido YouTubeForm
    if form.validate_on_submit():
        try:
            youtube_url = form.youtube_url.data
            transcription = whisper_util.process_youtube_video(youtube_url)

            # Guardar la transcripción en un archivo HTML
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            transcription_filename = f"transcription_{timestamp}.html"
            transcription_filepath = os.path.join('transcriptions', transcription_filename)
            with open(transcription_filepath, 'w') as transcription_file:
                transcription_file.write(transcription)
            
            flash('Transcripción completada con éxito.', 'success')
            # Pasar la ruta al archivo de transcripción en lugar de la transcripción directamente
            return render_template('transcription.html', form=form, transcription_filepath=transcription_filepath)
        except FileNotFoundError:
            flash('Error al procesar el video. Asegúrate de que el enlace sea correcto.', 'error')
        except Exception as e:
            flash(f'Ocurrió un error inesperado: {e}', 'error')
    return render_template('transcription.html', form=form, transcription_filepath=None)


@app.route('/upload_pdf', methods=['GET', 'POST'])
def upload_pdf():
    form = PDFUploadForm()
    if form.validate_on_submit():
        pdf_file = form.pdf.data
        filename = secure_filename(pdf_file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        pdf_path = os.path.join('path/to/uploaded_pdfs', unique_filename)
        pdf_file.save(pdf_path)
        text = extract_text_from_pdf(pdf_path)
        return redirect(url_for('chat', context=text))
    return render_template('upload_pdf.html', form=form)


@app.route('/chat_interface')
def chat_interface():
    # Lista los archivos en el directorio de transcripciones
    transcriptions = os.listdir('transcriptions')
    # Renderiza una plantilla pasando los nombres de los archivos de transcripción
    return render_template('chat.html', transcriptions=transcriptions)

# Asegúrate de reemplazar 'your_huggingface_api_token' con tu token real de la API de Hugging Face
YOUR_HUGGING_FACE_API_TOKEN = os.getenv("YOUR_HUGGING_FACE_API_TOKEN")
headers = {"Authorization": f"Bearer {YOUR_HUGGING_FACE_API_TOKEN}"}

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message')
    transcription_filename = data.get('transcription_filename')

    transcription_path = os.path.join('transcriptions', transcription_filename)
    with open(transcription_path, 'r') as file:
        transcription_text = file.read()

    payload = {
        "inputs": {
            "question": user_input,
            "context": transcription_text
        }
    }

    response = requests.post("https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1", headers=headers, json=payload)
    answer = response.json()

    return jsonify({'response': answer['answer']})

if __name__ == '__main__':
    app.run(debug=True)