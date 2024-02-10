from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from .forms import AudioUploadForm, YouTubeForm, PDFUploadForm  # Asegúrate de incluir PDFUploadForm
from werkzeug.utils import secure_filename  # Importación para manejar nombres de archivo seguros
from .util import whisper_util
import os
from app import app
from datetime import datetime
# Asegúrate de importar get_mistral_response correctamente
from .util.chatbot import get_mistral_response


@app.route('/')
def index():
    return render_template('base.html', now=datetime.now())

@app.route('/upload', methods=['GET', 'POST'])
def upload_audio():
    form = AudioUploadForm()
    if form.validate_on_submit():
        audio_file = form.audio.data
        audio_file.save(os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename))
        transcription = whisper_util.transcribe_audio(os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename))
        return render_template('transcription.html', transcription=transcription)
    return render_template('upload.html', form=form)



@app.route('/transcribe_youtube', methods=['GET', 'POST'])
def transcribe_youtube():
    form = YouTubeForm()  # Asumimos que ya tienes definido YouTubeForm
    if form.validate_on_submit():
        try:
            youtube_url = form.youtube_url.data
            transcription = whisper_util.process_youtube_video(youtube_url)
            flash('Transcripción completada con éxito.', 'success')
            return render_template('transcription.html', form=form, transcription=transcription)
        except FileNotFoundError:
            flash('Error al procesar el video. Asegúrate de que el enlace sea correcto.', 'error')
        except Exception as e:
            flash('Ocurrió un error inesperado. Por favor, inténtalo de nuevo.', 'error')
            # Considera loguear el error e para propósitos de depuración
    # Asegúrate de pasar el objeto form incluso si no se valida para mantener el formulario rellenable
    return render_template('transcription.html', form=form, transcription="Aquí va el texto de la transcripción obtenida")

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

@app.route('/chat', methods=['POST'])
def chat():
    # Acceder al JSON enviado en la solicitud
    data = request.json
    user_input = data.get('message') # Obtener el mensaje del usuario del JSON

    if user_input:
        response = get_mistral_response(user_input)
        return jsonify({'response': response})
    else:
        return jsonify({'response': 'No se proporcionó entrada válida.'})

if __name__ == '__main__':
    app.run(debug=True)


