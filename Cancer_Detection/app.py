import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import numpy as np

# Create the Flask app
app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for file uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff'}

# Load the trained model
model = load_model('model/model_history.keras')

def allowed_file(filename):
    """Check if the file has one of the allowed extensions"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(file_path):
    """Load and preprocess the image for prediction"""
    image = load_img(file_path, target_size=(96, 96))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def convert_to_png(filepath):
    """Convert .tif images to .png"""
    if filepath.lower().endswith('.tif') or filepath.lower().endswith('.tiff'):
        img = Image.open(filepath)
        png_filepath = filepath.rsplit('.', 1)[0] + '.png'
        img.save(png_filepath, 'PNG')
        return png_filepath
    return filepath

@app.route('/')
def index():
    return render_template('index.html', filename=None)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index', message="No file part"))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index', message="No selected file"))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Convert image to .png if necessary
        file_path = convert_to_png(file_path)
        filename = os.path.basename(file_path)
        
        # Prepare the image and make prediction
        image = prepare_image(file_path)
        prediction = model.predict(image)[0][0]
        

        # Determine the result
        if prediction > 0.5:
            result = f"There is a  %{prediction*100:.2f} probability that there is metastatic cancer in the uploaded tissue sample. Please intervene immediately."
        else:
            result = f"It was detected that there was no metastatic cancer in the uploaded tissue sample."
        
        return render_template('result.html', filename=filename, result=result)
    return redirect(url_for('index', message="Invalid file type"))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
    
