from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, current_app as app
from .forms import AudioUploadForm, YouTubeForm, PDFUploadForm  # Asegúrate de incluir PDFUploadForm
from werkzeug.utils import secure_filename  # Importación para manejar nombres de archivo seguros
from .util import whisper_util
import os
from .models import Usuario
from .util  import chatbot
from langchain_groq import ChatGroq
from app.extensions import db
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from transformers import pipeline 
from datetime import datetime
from langchain_community.docstore.document import Document
import requests
import logging
import uuid  # MODIFICADO: Asegúrate de importar uuid si vas a usarlo
from dotenv import load_dotenv
 # Asegúrate de haber instalado langchain
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader # Asume que has creado o adaptado una clase para cargar texto
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from app.forms import LoginForm, RegistroForm
from app.models import Usuario

@app.route('/registro', methods=['GET', 'POST'])
def registro():
    form = RegistroForm()
    if form.validate_on_submit():
        user = Usuario(nombre=form.nombre.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('¡Registro exitoso!')
        return redirect(url_for('login'))
    return render_template('registro.html', title='Registro', form=form)

from flask_login import login_user

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = Usuario.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember_me.data)
            flash('Inicio de sesión exitoso.', 'success')
            return redirect(url_for('base'))
        else:
            flash('Inicio de sesión fallido. Revisa tu correo y contraseña.', 'error')
    return render_template('login.html', title='Iniciar Sesión', form=form)

from flask_login import logout_user

@app.route('/logout')
def logout():
    logout_user()
    flash('Has cerrado sesión con éxito.', 'info')
    return redirect(url_for('index'))


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




import fitz  # Asegúrate de haber instalado PyMuPDF

@app.route('/upload_pdf', methods=['GET', 'POST'])
def upload_pdf():
    form = PDFUploadForm()
    if form.validate_on_submit():
        pdf_file = form.pdf.data
        filename = secure_filename(pdf_file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        pdf_file.save(pdf_path)

        # Leer el contenido del PDF subido
        pdf_content = read_pdf_content(pdf_path)
        flash('Archivo PDF cargado con éxito.', 'success')
        
        # Almacenar el contenido del PDF en la sesión
        session['pdf_content'] = pdf_content

        return redirect(url_for('generate_test_route'))
    return render_template('upload_pdf.html', form=form)


@app.route('/select_pdf')
def select_pdf():
    try:
        pdfs_dir = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
        if not os.path.exists(pdfs_dir):
            os.makedirs(pdfs_dir)  # Crea la carpeta si no existe
        pdf_files = [f for f in os.listdir(pdfs_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            # Si no hay archivos PDF, envía un mensaje al usuario
            flash('No se encontraron archivos PDF en la carpeta de subidas.', 'info')
    except Exception as e:
        # Captura cualquier excepción y envía un mensaje al usuario
        flash('Ocurrió un error al intentar listar los archivos PDF.', 'error')
        logging.error(f'Error al listar archivos PDF en {pdfs_dir}: {e}')
        pdf_files = []  # Asegúrate de que pdf_files esté definido aunque ocurra un error
    
    return render_template('select_pdf.html', pdf_files=pdf_files)


def read_pdf_content(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()  # Es una buena práctica cerrar el documento después de usarlo
        return text
    except fitz.fitz.FileNotFoundError:
        # Manejar específicamente el error cuando el archivo PDF no se encuentra
        logging.error(f"El archivo PDF en {pdf_path} no fue encontrado.")
        return None
    except fitz.fitz.PDFSyntaxError:
        # Manejar específicamente los errores de sintaxis en el archivo PDF
        logging.error(f"Error de sintaxis en el archivo PDF en {pdf_path}.")
        return None
    except Exception as e:
        # Capturar cualquier otra excepción que no se haya anticipado
        logging.error(f"Error al leer el contenido del PDF {pdf_path}: {e}")
        return None



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
        # Aquí es donde comienza a manejar la solicitud POST
        logging.info("Recibida solicitud POST a /chat_interface.")
        
        data = request.json
        question = data.get('question')
        
        # Agrega logs para ver qué datos se están recibiendo
        logging.info(f"Datos recibidos: {data}")
        logging.info(f"Pregunta recibida: {question}")

        if not question or not transcription_filename:
            # Si entra en esta condición, es probablemente la causa del error 400
            logging.error("Falta información de entrada o nombre de archivo de transcripción.")
            return jsonify({'error': 'Falta información de entrada o nombre de archivo de transcripción'}), 400

        transcription_path = os.path.join(transcriptions_dir, transcription_filename)
        logging.info(f"Ruta del archivo de transcripción a abrir: {transcription_path}")
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

        # Crear objetos Document a partir de los textos
        documents = [Document(page_content=text) for text in texts]

        # Indexar en Chroma
        db = Chroma.from_documents(documents, embeddings)
        retriever = db.as_retriever(k=2)

        # Usar ChatGroq
        chat = ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))

        # Crear cadena conversacional 
        qa_chain = ConversationalRetrievalChain.from_llm(chat, retriever, return_source_documents=True)

        # Realizar QA
        result = qa_chain({"question": question, "chat_history": chat_history})
        logging.info(f"Respuesta del chat: {result['answer']}")
        # Agregar al historial y guardar en la sesión
        chat_history.append((question, result['answer']))
        session['chat_history'] = chat_history

        return jsonify({'response': result['answer']})

    # La sección GET simplemente renderiza la plantilla con los archivos de transcripción
    return render_template('chat.html', transcription_files=transcription_files, transcription_filename=transcription_filename)


from langchain.prompts import PromptTemplate

@app.route('/generate_test', methods=['GET'])
def generate_test():
    pdf_content = session.get('pdf_content')
    if not pdf_content:
        flash('No hay contenido de PDF disponible para generar el test.', 'error')
        return redirect(url_for('index'))

    num_questions = request.args.get('num_questions', default=3, type=int)

    chat = ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"), language="es")

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

    session['questions'] = questions  # Almacena las preguntas en la sesión
    return redirect(url_for('ask_questions'))


@app.route('/generate_test_route')
def generate_test_route():
    # Obtener el contenido del PDF desde la sesión
    pdf_content = session.get('pdf_content')
    if not pdf_content:
        flash('No hay contenido de PDF disponible para generar el test.', 'error')
        return redirect(url_for('base'))

    num_questions = 5  # O cualquier otro valor predeterminado que desees

    # Redirigir a la ruta que realmente genera el test
    return redirect(url_for('generate_test', num_questions=num_questions))


def check_answer(question_text, user_answer, options, pdf_content):
    chat = ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))
    system = """Eres un asistente que evalúa respuestas de opción múltiple. Dada la pregunta, la respuesta del usuario, las opciones y el contenido del PDF, determina si la respuesta del usuario es correcta basándote en la información del PDF. Responde con 'sí' o 'no'."""

    human = f"""Contenido del PDF:
    {pdf_content}

    Pregunta: {question_text}

    Opciones:
    {[option for option in options]}

    Respuesta del usuario: {user_answer}"""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", human)]
    )
    print(prompt.format_prompt(question_text=question_text, user_answer=user_answer, options=options, pdf_content=pdf_content))
    response = prompt | chat
    is_correct = response.invoke({}).content.lower().startswith("sí")
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
        user_answers = []
        for i in range(total_questions):
            user_answer = request.form.get(f"question{i}")
            user_answers.append(user_answer)

        pdf_content = session.get('pdf_content')  # Obtener el contenido del PDF desde la sesión

        for i, question in enumerate(questions):
            user_answer = user_answers[i]
            options = question["choices"]
            question_text = question["question"]
            correctness = check_answer(question_text, user_answer, options, pdf_content)
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
    
@app.route('/submit_test', methods=['POST'])
def submit_test():
    questions = session.get('questions', [])
    score = 0
    total_questions = len(questions)
    user_answers = []

    for i in range(total_questions):
        # Este es el cambio clave: Asegúrate de que se recoge correctamente la respuesta de cada pregunta basada en su índice
        user_answer = request.form.get(f"question{i}")
        user_answers.append(user_answer)

    pdf_content = session.get('pdf_content')

    for i, question in enumerate(questions):
        user_answer = user_answers[i]
        options = question["choices"]
        question_text = question["question"]
        correctness = check_answer(question_text, user_answer, options, pdf_content)
        if correctness == "correct":
            score += 1

    flash(f'Tu puntuación es {score}/{total_questions}', 'info')
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)

