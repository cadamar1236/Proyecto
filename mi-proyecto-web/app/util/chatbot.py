import requests
import os
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

def chat_with_pdf(pdf_path, query):
    # Cargar PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Dividir texto
    splitter = CharacterTextSplitter(chunk_size=1000)
    texts = splitter.split_documents(documents)

    # Obtener embeddings
    embeddings = HuggingFaceEmbeddings()

    # Indexar en Chroma
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever(k=2)

    # Usar ChatGroq en lugar de HuggingFaceHub
    chat = ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))

    # Crear cadena conversacional
    qa_chain = ConversationalRetrievalChain.from_llm(chat, retriever, return_source_documents=True)

    # Preguntar al chatbot
    result = qa_chain({"question": query, "chat_history": []})
    return result["answer"]

