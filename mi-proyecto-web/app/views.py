from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from .forms import AudioUploadForm, YouTubeForm, PDFUploadForm  # Asegúrate de incluir PDFUploadForm
from werkzeug.utils import secure_filename  # Importación para manejar nombres de archivo seguros
from .util import whisper_util
import os
from chatbot import chat_with_pdf 
from app import app
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from transformers import pipeline 
from datetime import datetime
import requests
import uuid  # MODIFICADO: Asegúrate de importar uuid si vas a usarlo
from dotenv import load_dotenv
 # Asegúrate de haber instalado langchain
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader  # Asume que has creado o adaptado una clase para cargar texto
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.retrievers import BaseRetriever
from langchain.chains import ConversationalRetrievalChain


load_dotenv()
token = os.getenv("YOUR_HUGGING_FACE_API_TOKEN")
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
        pdf_path = os.path.join('uploads', filename)
        pdf_file.save(pdf_path)

        # Aquí simplemente redirigimos al usuario a la ruta de chat con el nombre del archivo como argumento
        return redirect(url_for('chat_with_pdf_context', pdf_path=filename))
    return render_template('upload_pdf.html', form=form)

@app.route('/chat_with_pdf_context', methods=['GET', 'POST'])
def chat_with_pdf_context():
    pdf_path = request.args.get('pdf_path', '')

    if request.method == 'POST':
        question = request.form['message']
        # Asegúrate de construir el camino completo al archivo PDF aquí
        full_pdf_path = os.path.join('uploads', pdf_path)
        response = chat_with_pdf(full_pdf_path, question)
        return render_template('chat.html', pdf_path=pdf_path, response=response)

    # Si es una solicitud GET, simplemente mostramos la interfaz de chat sin respuesta
    return render_template('chat.html', pdf_path=pdf_path)

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


# Asegúrate de haber configurado la autenticación con Hugging Face
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    transcription_filename = data.get('transcription_filename')

    if not question or not transcription_filename:
        return jsonify({'error': 'Missing input information or transcription filename'}), 400

    transcription_path = os.path.join('transcriptions', transcription_filename)
    try:
        with open(transcription_path, 'r') as file:
            transcription_text = file.read()
    except FileNotFoundError:
        return jsonify({'error': 'Transcription file not found'}), 404

    # Inicializar historial de chat
    chat_history = []
    
    # Dividir texto
    text_splitter = CharacterTextSplitter(chunk_size=1000)
    texts = text_splitter.split_text(transcription_text)

    # Obtener embeddings
    embeddings = HuggingFaceEmbeddings()

    # Indexar en Chroma
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever(k=2)

    # Usar ChatGroq
    chat = ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))

    # Crear cadena conversacional 
    qa_chain = ConversationalRetrievalChain.from_llm(chat, retriever, return_source_documents=True)

    # Realizar QA
    result = qa_chain({"question": question, "chat_history": chat_history})
    
    # Agregar al historial
    chat_history.append((question, result['answer']))

    return jsonify({'response': result['answer']})

if __name__ == '__main__':
    app.run(debug=True)

