from flask import Flask
from datetime import datetime
import os
from app.extensions import db  # Asegúrate de que extensions.py esté configurado correctamente

def create_app():
    # Inicializa la aplicación Flask
    app = Flask(__name__)
    
    # Configuración de la aplicación
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
    app.config['SECRET_KEY'] = 'edtech1245'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///miaplicacion.db'  # Añade la URI de la base de datos SQLite
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Desactiva la señalización

    db.init_app(app)  # Inicializa la extensión db con la aplicación Flask

    @app.context_processor
    def inject_now():
        return {'now': datetime.now()}
    
    # Deja estas importaciones aquí para evitar importaciones circulares
    with app.app_context():
        from . import views, models
    
    return app





