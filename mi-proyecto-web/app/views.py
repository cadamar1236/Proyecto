from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from .forms import AudioUploadForm, YouTubeForm, PDFUploadForm  # Asegúrate de incluir PDFUploadForm
from werkzeug.utils import secure_filename  # Importación para manejar nombres de archivo seguros
from .util import whisper_util
import os
from .util  import chatbot
from app import app
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from transformers import pipeline 
from datetime import datetime
import requests
import uuid  # MODIFICADO: Asegúrate de importar uuid si vas a usarlo
from dotenv import load_dotenv
 # Asegúrate de haber instalado langchain
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader # Asume que has creado o adaptado una clase para cargar texto
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
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

import fitz  # Asegúrate de haber instalado PyMuPDF

@app.route('/upload_pdf', methods=['GET', 'POST'])
def upload_pdf1():
    form = PDFUploadForm()
    if form.validate_on_submit():
        pdf_file = form.pdf.data
        filename = secure_filename(pdf_file.filename)
        pdf_path = os.path.join('uploads', filename)
        pdf_file.save(pdf_path)

        # Leer el contenido del PDF subido
        pdf_content = read_pdf_content(pdf_path)

        # Redirige con el contenido del PDF a la siguiente etapa
        session['pdf_content'] = pdf_content  # Almacenamos el contenido en la sesión
        return redirect(url_for('generate_test_route'))
    return render_template('upload_pdf.html', form=form)

@app.route('/select_pdf')
def select_pdf():
    pdfs_dir = os.path.join(app.root_path, 'uploads')
    pdf_files = [f for f in os.listdir(pdfs_dir) if f.endswith('.pdf')]
    
    return render_template('select_pdf.html', pdf_files=pdf_files)

def read_pdf_content(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


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

@app.route('/chat_interface', defaults={'transcription_filename': None}, methods=['GET', 'POST'])
@app.route('/chat_interface/<transcription_filename>', methods=['GET', 'POST'])
def chat_interface(transcription_filename):
    transcriptions_dir = os.path.join(app.root_path, 'transcriptions')
    transcription_files = [f for f in os.listdir(transcriptions_dir) if os.path.isfile(os.path.join(transcriptions_dir, f))]
    
    if request.method == 'POST':
        data = request.json
        question = data.get('question')

        if not question or not transcription_filename:
            return jsonify({'error': 'Falta información de entrada o nombre de archivo de transcripción'}), 400

        transcription_path = os.path.join(transcriptions_dir, transcription_filename)
        try:
            with open(transcription_path, 'r') as file:
                transcription_text = file.read()
        except FileNotFoundError:
            return jsonify({'error': 'Archivo de transcripción no encontrado'}), 404

        # Aquí se asume que el resto de tu lógica de chat está correcta y funciona como se espera
        # Inicializar historial de chat
        chat_history = session.get('chat_history', [])

        # Proceso de chat (dividir texto, obtener embeddings, indexar en Chroma, usar ChatGroq, crear cadena conversacional y realizar QA)
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
        
        # Agregar al historial y guardar en la sesión
        chat_history.append((question, result['answer']))
        session['chat_history'] = chat_history

        return jsonify({'response': result['answer']})

    # La sección GET simplemente renderiza la plantilla con los archivos de transcripción
    return render_template('chat.html', transcription_files=transcription_files, transcription_filename=transcription_filename)


from langchain.prompts import PromptTemplate

@app.route('/generate_test')
def generate_test(pdf_content, num_questions):

    chat = ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))

    system = """Eres un asistente que genera preguntas de opción múltiple. Debes proporcionar 3 preguntas sobre el texto dado, con 4 opciones de respuesta cada una. Usa el siguiente formato: 

    Pregunta 1: ¿Cuál es la capital de Francia?
    - París  
    - Londres
    - Berlín
    - Madrid"""  

    human = f"""Contenido del PDF:
    {pdf_content}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system), 
        ("human", human)
    ])
    input = {"pdf_content": pdf_content}

    response = prompt | chat
    response_msg = response.invoke(input)
    response_text = response_msg.content

    # Extraer preguntas
    questions = []
    for qa in response_text.split("\n\n"):
        lines = qa.split("\n")
        question = lines[0]
        choices = lines[1:]

        questions.append({
            "question": question,
            "choices": choices  
        })

    return questions

@app.route('/generate_test_route')
def generate_test_route():
    pdf_content = session.get('pdf_content', '')
    if not pdf_content:
        flash('No hay contenido de PDF disponible para generar el test.', 'error')
        return redirect(url_for('index'))

    num_questions = 5  # Definir el número de preguntas que quieres generar
    
    questions = generate_test(pdf_content, num_questions)
    session['questions'] = questions  # Almacenamos las preguntas en la sesión
    
    return redirect(url_for('take_test'))

def check_answer(question, user_answer, chat):
    chat = ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))
    system = """Eres un asistente que evalúa respuestas de opción múltiple. Dada la pregunta, la respuesta del usuario y las opciones, determina si la respuesta del usuario es correcta.Responde con si o no"""

    options = "".join("- " + choice + "\n" for choice in question["choices"])

    human = f"""Pregunta: {question["question"]}  
    Respuesta del usuario: {user_answer}
    Opciones:
    {options}"""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", human)]
    )

    response = prompt | chat
    print(response)
    print(response.invoke({}).content)
    is_correct = response.invoke({}).content.lower().startswith("sí")
    print(is_correct)
    if is_correct:
        return "correct"
    else: 
        return "incorrect"



@app.route('/take_test', methods=['GET', 'POST'])
def ask_questions():
    if request.method == 'POST':
        # Esta parte maneja las respuestas enviadas por el usuario
        questions = session.get('questions', [])
        score = 0
        total_questions = len(questions)
        for question in questions:
            user_answer = request.form.get(question["question"])
            correctness = check_answer(question, user_answer)
            if correctness == "correct":
                score += 1
        
        flash(f'Tu puntuación es {score}/{total_questions}', 'info')
        return redirect(url_for('index'))  # Redireccionar al inicio o a una página de resultados
    else:
        # Mostrar las preguntas si es una solicitud GET
        questions = session.get('questions', [])
        if not questions:
            # Redirigir al usuario si no hay preguntas en la sesión
            flash('No hay test disponible para realizar.', 'warning')
            return redirect(url_for('index'))
        return render_template('take_test.html', questions=questions)
    
if __name__ == '__main__':
    app.run(debug=True)

