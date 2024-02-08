# views.py
from flask import Flask, render_template, request, redirect, url_for
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
    form = YouTubeForm()
    if form.validate_on_submit():
        try:
            youtube_url = form.youtube_url.data
            transcription = whisper_util.process_youtube_video(youtube_url)
            return render_template('transcription.html', transcription=transcription)
        except FileNotFoundError:
            Flasklash('Error al procesar el video. Asegúrate de que el enlace sea correcto.', 'error')
        except Exception as e:
            Flash('Ocurrió un error inesperado. Por favor, inténtalo de nuevo.', 'error')
            # Log the exception e for debugging purposes
    return render_template('youtube_transcription.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)


