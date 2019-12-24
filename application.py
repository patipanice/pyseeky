from flask import Flask, request, jsonify ,render_template ,redirect ,url_for ,send_from_directory
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
import os 
import requests
import os.path as path
import cv2
import numpy as np
from random import shuffle
from keras.models import load_model
import imageio
import tensorflow as tf
modelpath ='modelreal.h5'
TRAIN_DIR = 'data/pt'
IMAGE_UPLOADS ="/"

app = Flask(__name__)
# Enable CORS
CORS(app)
app.config["IMAGE_UPLOADS"] ='data/pt/'
IMG_SIZE = 100
data = ['hat', 'headphone', 'laptop','bag','handbag','wallet','watch']
img = '1_eiei.png'
def create_label(image_name):
    word_label = image_name.split('_',1) 
    if word_label[1] == 'hat.png':
        return np.array([1,0,0,0,0,0,0])
    elif word_label[1] == 'headphone.png':
        return np.array([0,1,0,0,0,0,0])
    elif word_label[1] == 'laptop.png':
        return np.array([0,0,1,0,0,0,0])
    elif word_label[1] == 'bag.png':
        return np.array([0,0,0,1,0,0,0])
    elif word_label[1] == 'handbag.png':
        return np.array([0,0,0,0,1,0,0])
    elif word_label[1] == 'wallet.png':
        return np.array([0,0,0,0,0,1,0])
    elif word_label[1] == 'watch.png':
        return np.array([0,0,0,0,0,0,1])
def create_train_data():
	training_data = []
	path = os.path.join(TRAIN_DIR,img)
	img_data=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
	img_data=cv2.resize(img_data,(IMG_SIZE,IMG_SIZE))
	training_data.append([np.array(img_data),create_label(img)])
	shuffle(training_data)
	return training_data

@app.route("/predict", methods=["POST"])
def predict():
	if request.method == "POST":
		picinput = request.files['img']
		picinput.save(os.path.join(app.config["IMAGE_UPLOADS"],img))	
		train_data = create_train_data()
		train = train_data[:1]
		
		X_train = np.array([i[0] for i in train]).reshape(-1,100,100,1)
		Y_train = np.array([i[1] for i in train])
		
		load_naja =  load_model(modelpath)
		predicted = load_naja.predict(X_train)
		predicted 
		predicteds =np.argmax(predicted)
		print(data[predicteds])
		return data[predicteds]
		
# run the app.
#if __name__ == "__main__":
#	app.run()
