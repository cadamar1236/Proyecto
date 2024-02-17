from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from .forms import AudioUploadForm, YouTubeForm, PDFUploadForm  # Asegúrate de incluir PDFUploadForm
from werkzeug.utils import secure_filename  # Importación para manejar nombres de archivo seguros
from .util import whisper_util
import os
from app import app
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from datetime import datetime
import requests
import uuid  # MODIFICADO: Asegúrate de importar uuid si vas a usarlo

@app.route('/')
def index():
    return render_template('base.html', now=datetime.now())

@app.route('/upload_audio', methods=['GET', 'POST'])
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
        
        flash('Transcripción completada con éxito.', 'success')
        return redirect(url_for('chat_interface', transcription_filename=transcription_filename))
    
    return render_template('upload.html', form=form)

@app.route('/transcribe_youtube', methods=['GET', 'POST'])
def transcribe_youtube():
    form = YouTubeForm()
    if form.validate_on_submit():
        try:
            youtube_url = form.youtube_url.data
            transcription = whisper_util.process_youtube_video(youtube_url)

            # Guardar la transcripción en un archivo
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            transcription_filename = f"transcription_{timestamp}.html"
            transcription_filepath = os.path.join('transcriptions', transcription_filename)
            with open(transcription_filepath, 'w') as transcription_file:
                transcription_file.write(transcription)
            
            flash('Transcripción completada con éxito.', 'success')
            return redirect(url_for('chat_interface', transcription_filename=transcription_filename))
        except FileNotFoundError:
            flash('Error al procesar el video. Asegúrate de que el enlace sea correcto.', 'error')
        except Exception as e:
            flash(f'Ocurrió un error inesperado: {e}', 'error')
    return render_template('transcription.html', form=form)

@app.route('/upload_pdf', methods=['GET', 'POST'])
def upload_pdf():
    form = PDFUploadForm()
    if form.validate_on_submit():
        pdf_file = form.pdf.data
        filename = secure_filename(pdf_file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"  # MODIFICADO: Asegúrate de que esta línea se corresponde con tu implementación
        pdf_path = os.path.join('path/to/uploaded_pdfs', unique_filename)
        pdf_file.save(pdf_path)
        text = extract_text_from_pdf(pdf_path)  # MODIFICADO: Asegúrate de implementar esta función
        return redirect(url_for('chat', context=text))
    return render_template('upload_pdf.html', form=form)

@app.route('/chat_interface', defaults={'transcription_filename': None})
@app.route('/chat_interface/<transcription_filename>')
def chat_interface(transcription_filename):
    transcriptions_dir = os.path.join(app.root_path, 'transcriptions')
    transcription_files = [f for f in os.listdir(transcriptions_dir) if os.path.isfile(os.path.join(transcriptions_dir, f))]
    
    transcription_text = None
    if transcription_filename:
        transcription_path = os.path.join(transcriptions_dir, transcription_filename)
        if os.path.exists(transcription_path):
            with open(transcription_path, 'r') as file:
                transcription_text = file.read()
        else:
            flash('Transcripción no encontrada.', 'error')
    
    return render_template('chat.html', transcription_files=transcription_files, transcription_text=transcription_text)

# Configuración del Modelo Llama 2
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
pipeline = pipeline("text-generation", model=MODEL_NAME, device=0)  # Asume que estás utilizando una GPU

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('message')
    transcription_filename = data.get('transcription_filename')

    if not question or not transcription_filename:
        return jsonify({'error': 'Falta información de entrada o nombre de archivo de transcripción'}), 400

    transcription_path = os.path.join('transcriptions', transcription_filename)
    try:
        with open(transcription_path, 'r') as file:
            context = file.read()
    except FileNotFoundError:
        return jsonify({'error': 'Archivo de transcripción no encontrado'}), 404

    # Generar respuesta utilizando el pipeline y el contexto de la transcripción
    response = pipeline(f"{context}\n\n{question}", max_length=512, clean_up_tokenization_spaces=True)

    # Extraer y devolver la respuesta generada
    answer = response[0]['generated_text'][len(context):]  # Ajusta según sea necesario para limpiar la salida
    return jsonify({'response': answer})


if __name__ == '__main__':
    app.run(debug=True)

