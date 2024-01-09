from flask import Flask, render_template, request, redirect, url_for, flash, session
 
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField, FileField, BooleanField, HiddenField, SelectMultipleField
from wtforms.widgets import ListWidget, CheckboxInput
from wtforms.validators import DataRequired
import pandas as pd
from supervised.automl import AutoML
from bs4 import BeautifulSoup
from ydata_profiling import ProfileReport
import os 

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'
app.config['UPLOAD_FOLDER'] = './uploads'

# Global variable to store the DataFrame
uploaded_dataframe = None


class DataUploadForm(FlaskForm):
    file = FileField('Upload CSV with training data', validators=[DataRequired()])
    selected_columns = BooleanField('Select Columns')
    submit = SubmitField('Save')

class ModelTrainingForm(FlaskForm):
    x_columns = StringField('Input features (comma-separated)', validators=[DataRequired()])
    y_column = StringField('Target column', validators=[DataRequired()])
    mode = SelectField('AutoML Mode', choices=[('Explain', 'Explain'), ('Perform', 'Perform'), ('Compete', 'Compete')])
    algorithms = StringField('Algorithms (comma-separated)', validators=[DataRequired()])
    time_limit = SelectField('Time limit (seconds)',
                            choices=[('60', '60'), ('120', '120'), ('240', '240'), ('300', '300')])
    submit = SubmitField('Start Training')

 


# adding username and password for logging in
allowed_usernames = ['hassan', 'obai', 'sozana', 'laith', 'madeleine']

@app.route('/')   
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    if username in allowed_usernames and password == '1234':
        return redirect('/home')
    else:
        error_message = 'error'
        if request.method == 'POST':
            if 'reset' in request.form:
                return redirect('/home')
        error_message = 'The user name or password is incorrect'
        flash(error_message)
        return render_template('login.html', error_message=error_message)
 
@app.route('/home', methods=['GET', 'POST'])
def home():
    global uploaded_dataframe
    form = DataUploadForm()
    display_table = False

    if form.validate_on_submit():
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        # Check if the "OK" button is clicked
        if request.method == 'POST' and 'ok_button' in request.form:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], form.file.data.filename)
            form.file.data.save(filepath)
            uploaded_dataframe = pd.read_csv(filepath)
            session['uploaded_dataframe'] = uploaded_dataframe.to_json(orient='split')
            display_table = True

   

        # Check if the "Perform EDA" button is clicked
        if request.method == 'POST' and 'perform_eda' in request.form:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], form.file.data.filename)
            form.file.data.save(filepath)
            uploaded_dataframe = pd.read_csv(filepath)
            profile = ProfileReport(uploaded_dataframe, title="Profiling Report")
            profile_html = profile.to_html()
            return render_template('eda_results.html', vis=profile_html)

    # Check if the DataFrame is available in the session and load it
    if 'uploaded_dataframe' in session:
        uploaded_dataframe = pd.read_json(session['uploaded_dataframe'], orient='split')

    return render_template('home.html', form=form, uploaded_dataframe=uploaded_dataframe, display_table=display_table)

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    global uploaded_dataframe
    form = ModelTrainingForm()

    if form.validate_on_submit():
        algorithms = form.algorithms.data.split(',')
        filepath = request.args.get('filepath')
        
         
        automl = AutoML(mode=form.mode.data, algorithms=form.algorithms.data.split(','),
                        total_time_limit=int(form.time_limit.data))
        automl.fit(uploaded_dataframe[form.x_columns.data.split(',')], uploaded_dataframe[form.y_column.data])
        html_content_data = automl.report().data
        soup = BeautifulSoup(html_content_data, 'html.parser')
        image_src_to_remove = "https://raw.githubusercontent.com/mljar/visual-identity/main/media/mljar_AutomatedML.png"
        image_tags_to_remove = soup.find_all('img', src=image_src_to_remove)
        for img_tag in image_tags_to_remove:
            img_tag.extract() 
        edited_html_content = str(soup)

        return render_template('results.html', automl=automl, edited_html_content=edited_html_content)
    return render_template('train.html', form=form)
if __name__ == "__main__":
    app.run(debug=True)
