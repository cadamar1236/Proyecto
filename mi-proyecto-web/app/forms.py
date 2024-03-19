# forms.py
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, URL

class RegistroForm(FlaskForm):
    nombre = StringField('Nombre', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Contraseña', validators=[DataRequired()])
    password2 = PasswordField('Repite Contraseña', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Registrar')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Contraseña', validators=[DataRequired()])
    remember_me = BooleanField('Recuérdame')
    submit = SubmitField('Iniciar Sesión')

class AudioUploadForm(FlaskForm):
    audio = FileField('Upload Audio', validators=[FileAllowed(['mp3', 'wav', 'ogg'], 'Audio only!')])
    submit = SubmitField('Transcribe')

class YouTubeForm(FlaskForm):
    youtube_url = StringField('YouTube URL', validators=[DataRequired(), URL()])
    submit = SubmitField('Transcribe YouTube Video')

class PDFUploadForm(FlaskForm):
    pdf = FileField('Subir PDF', validators=[FileAllowed(['pdf'], 'Solo PDFs')])
    submit = SubmitField('Procesar')
