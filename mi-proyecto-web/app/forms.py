# forms.py
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, URL

class AudioUploadForm(FlaskForm):
    audio = FileField('Upload Audio', validators=[FileAllowed(['mp3', 'wav', 'ogg'], 'Audio only!')])
    submit = SubmitField('Transcribe')

class YouTubeForm(FlaskForm):
    youtube_url = StringField('YouTube URL', validators=[DataRequired(), URL()])
    submit = SubmitField('Transcribe YouTube Video')

class PDFUploadForm(FlaskForm):
    pdf = FileField('Subir PDF', validators=[FileAllowed(['pdf'], 'Solo PDFs')])
    submit = SubmitField('Procesar')
