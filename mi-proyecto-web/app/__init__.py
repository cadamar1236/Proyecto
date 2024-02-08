from flask import Flask
from datetime import datetime
# Inicializa la aplicación Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'C:\\Users\\User\\Desktop\\Proyecto\\Proyecto'
# Establece una clave secreta para tu aplicación
app.config['SECRET_KEY'] = 'edtech1245'
@app.context_processor
def inject_now():
    return {'now': datetime.now()}
from app import views



