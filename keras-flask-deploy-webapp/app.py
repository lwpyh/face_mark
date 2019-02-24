from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import pickle
import face_recognition as fr

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/face_rank_model.h5'

# Load your trained model
# model=make_network()
# model.load_weights (MODEL_PATH)
model=load_model(MODEL_PATH)
model.summary()
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet') #zhushidiaole
# print('Model loaded. Check http://127.0.0.1:5000/') #zhushidiaole


def model_predict(img_path, model):

    # Preprocessing the image
    image = fr.load_image_file(img_path)
    encs = fr.face_encodings(image)
    # if len(encs) != 1:
    #     print("Find %d faces in %s" % (len(encs), img_path))
    #     continue

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')  wozhushidiaole

    preds = model.predict(np.array(encs))
    print(type(preds))
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        # preds=model_predict("C://l/c1.jpg",model)
        print(preds[0])
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        t=round(preds[0][0]*2,3)
        print('t',t)
        print(type(t))
        # result=t.tolist()
        # print(result)
        # print(type(result))
        result=str(t)
        print(result)
        result1 = str(result)
        return result1
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()

