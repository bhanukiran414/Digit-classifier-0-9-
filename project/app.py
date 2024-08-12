from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
model = load_model('model.h5')

def prepare_image(img):
    img = img.convert('L') 
    img = img.resize((28, 28)) 
    img_array = np.array(img)  
    img_array = img_array.reshape((1, 28, 28, 1)) 
    img_array = img_array.astype('float32') / 255 
    return img_array

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file provided')
    
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    img_array = prepare_image(img)
    
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    
    return render_template('index.html', result=int(predicted_digit))

if __name__ == '__main__':
    app.run(debug=True)
