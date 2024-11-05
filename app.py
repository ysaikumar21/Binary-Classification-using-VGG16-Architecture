import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import tensorflow.keras
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from PIL import Image
app = Flask(__name__)
# Configure upload folder
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the saved model
with open('Binary_cnn_Vgg16.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_image(image_path):
    classes = {0: 'Cat', 1: 'Dog'}
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    prediction = np.argmax(prediction, axis=1)
    prediction = classes[prediction[0]]
    return prediction

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:

        return 'No image file'

    file = request.files['image']
    if file.filename == '':
        return 'No image selected for uploading'

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    prediction = predict_image(file_path)


    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)