# views.py
from flask import Flask, render_template, request, redirect, url_for, flash
from .forms import AudioUploadForm, YouTubeForm
from .util import whisper_util
import os
from app import app
from datetime import datetime


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


if __name__ == '__main__':
    app.run(debug=True)


