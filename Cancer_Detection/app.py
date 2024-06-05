# from flask import Flask, render_template, request, redirect, url_for
# import os
# from werkzeug.utils import secure_filename
# from keras.models import load_model
# from keras.preprocessing import image
# import numpy as np

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff'}

# model = load_model('/model/model_history.keras')

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def predict_image(filepath):
#     img = image.load_img(filepath, target_size=(96, 96))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0
#     prediction = model.predict(img_array)
#     return prediction[0][0]

# @app.route('/')
# def index():
#     return render_template('/templates/index.html')

# @app.route('/predict', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         prediction = predict_image(filepath)
#         result = 'Kanserli' if prediction > 0.5 else 'Normal'
#         return redirect(url_for('result', filename=filename, prediction=prediction, result=result))
#     return redirect(request.url)

# @app.route('/result')
# def result():
#     filename = request.args.get('filename')
#     prediction = request.args.get('prediction')
#     result = request.args.get('result')
#     return render_template('/templates/result.html', filename=filename, prediction=prediction, result=result)

# if __name__ == '__main__':
#     app.run(debug=True)
    
# from flask import Flask, render_template, request, redirect, url_for
# import os
# from werkzeug.utils import secure_filename
# from keras.models import load_model
# from keras.preprocessing import image
# import numpy as np

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = '/uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff'}

# model = load_model('/model/model_history.keras')

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def predict_image(filepath):
#     img = image.load_img(filepath, target_size=(96, 96))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0
#     prediction = model.predict(img_array)
#     return prediction[0][0]

# @app.route('/', methods=['GET'])
# def index():
#     return render_template('/templates/index.html')

# @app.route('/predict', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         prediction = predict_image(filepath)
#         result = 'Kanserli' if prediction > 0.5 else 'Normal'
#         return redirect(url_for('result', filename=filename, prediction=prediction, result=result))
#     return redirect(request.url)

# @app.route('/result', methods=['GET'])
# def result():
#     filename = request.args.get('filename')
#     prediction = request.args.get('prediction')
#     result = request.args.get('result')
#     return render_template('/templates/result.html', filename=filename, prediction=prediction, result=result)

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, render_template, request, jsonify
# import os
# from werkzeug.utils import secure_filename
# from keras.models import load_model
# from keras.preprocessing import image
# import numpy as np

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = '/uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff'}

# model = load_model('/model/model_history.keras')

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def predict_image(filepath):
#     img = image.load_img(filepath, target_size=(96, 96))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0
#     prediction = model.predict(img_array)
#     return prediction[0][0]

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         prediction = predict_image(filepath)
#         result = 'Kanserli' if prediction > 0.5 else 'Normal'
#         return jsonify({'filename': filename, 'prediction': float(prediction), 'result': result})
#     return jsonify({'error': 'File not allowed'})

# if __name__ == '__main__':
#     if not os.path.exists(app.config['UPLOAD_FOLDER']):
#         os.makedirs(app.config['UPLOAD_FOLDER'])
#     app.run(debug=True)

# /CANCER_APP/app.py

# from flask import Flask, request, redirect, url_for, render_template
# from werkzeug.utils import secure_filename
# import tensorflow
# import os
# from keras.models import load_model
# from keras.preprocessing.image import  img_to_array, load_img
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import numpy as np

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads/'
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# # Load model
# model = load_model('model/model_history.keras')

# # Ensure upload folder exists
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         # Process image
#         image = load_img(filepath, target_size=(96, 96))
#         image = img_to_array(image)
#         image = np.expand_dims(image, axis=0) / 255.0
        
#         # Predict
#         prediction = model.predict(image)
#         prediction_label = 'Cancerous' if prediction[0][0] > 0.5 else 'Normal'
#         prediction_confidence = prediction[0][0]

#         return render_template('result.html', filename=filename, label=prediction_label, confidence=prediction_confidence)

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return redirect(url_for('/statics', filename='uploads/' + filename), code=301)

# if __name__ == '__main__':
#     app.run(debug=True)
    
    

# from flask import Flask, render_template, request, redirect, url_for, send_from_directory
# from keras.models import load_model
# from werkzeug.utils import secure_filename
# import os
# import numpy as np
# from PIL import Image

# app = Flask(__name__, static_url_path='/statics')
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff'}
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size: 16MB

# # Load model
# model = load_model('model/model_history.keras')

# # Function to check allowed file extensions
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# # Function to process image and make prediction
# def predict_image(filepath):
#     # Image preprocessing
#     img = Image.open(filepath)
#     img = img.resize((96, 96))
#     img = np.array(img) / 255.0
#     img = np.expand_dims(img, axis=0)
    
#     # Prediction
#     prediction = model.predict(img)
#     return prediction[0][0]

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         # Make prediction
#         prediction = predict_image(filepath)
#         result = 'Cancerous' if prediction > 0.5 else 'Normal'
#         probability = prediction
        
#         return render_template('result.html', result=result, probability=probability, filepath=filepath)
#     return redirect(request.url)

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(debug=True)




# from flask import Flask, render_template, request, redirect, url_for
# import tensorflow
# import os
# from keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from werkzeug.utils import secure_filename
# import numpy as np
# import matplotlib.pyplot as plt

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff'}

# model = load_model('model/model_history.keras')

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST', 'GET'])
# def predict():
#     if 'file' not in request.files:
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         prediction, probability = model_predict(filepath, model)
#         return render_template('result.html', filename=filename, prediction=prediction, probability=probability)
#     return redirect(request.url)

# def model_predict(img_path, model):
#     datagen = ImageDataGenerator(rescale=1./255)
#     img = datagen.flow_from_directory('uploads', target_size=(96, 96), batch_size=1, class_mode=None, shuffle=False)
#     pred = model.predict(img)
#     pred_class = 'Cancerous' if pred[0][0] > 0.5 else 'Normal'
#     return pred_class, pred[0][0]

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return redirect(url_for('static', filename='uploads/' + filename), code=301)

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, render_template, request, redirect, url_for
# from werkzeug.utils import secure_filename
# import os
# from keras.models import load_model
# import numpy as np
# from tensorflow.keras.preprocessing import image
# import matplotlib.pyplot as plt
# import io
# import base64

# app = Flask(__name__)

# # Configuration
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Load model
# model = load_model('model/model_history.keras')

# def model_predict(img_path, model):
#     img = image.load_img(img_path, target_size=(96, 96))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = x / 255.0
#     preds = model.predict(x)
#     return preds

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/contact')
# def contact():
#     return render_template('contact.html')

# @app.route('/predict', methods=['POST'])
# def upload():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(file_path)

#             # Make prediction
#             preds = model_predict(file_path, model)
#             result = preds[0][0]

#             # Prepare image for display
#             img = image.load_img(file_path, target_size=(96, 96))
#             img_array = image.img_to_array(img)
#             img_array = img_array / 255.0

#             # Generate plot
#             fig, ax = plt.subplots()
#             ax.imshow(img_array)
#             ax.axis('off')
#             title = "Kanserli" if result > 0.5 else "Normal"
#             ax.set_title(f"Tahmin: {title} [{result:.4f}]")
            
#             # Convert plot to base64 string
#             buf = io.BytesIO()
#             plt.savefig(buf, format="png")
#             buf.seek(0)
#             image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
#             plt.close(fig)

#             return render_template('result.html', prediction=title, probability=result, image_base64=image_base64)
#     return redirect(url_for('result'))

# if __name__ == '__main__':
#     app.run(debug=True)
    
    

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
    
