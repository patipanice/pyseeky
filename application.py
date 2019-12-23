from flask import Flask, request, jsonify ,render_template ,redirect ,url_for ,send_from_directory
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
import os 
import requests
#import input_image # import ai มาใช้ในไฟล์นี้
import os.path as path
import cv2
#import pandas as pd
#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#from tqdm import tqdm
import numpy as np
from random import shuffle
from keras.models import load_model

import imageio

import tensorflow as tf
modelpath ='modelreal.h5'
TRAIN_DIR = 'data/pt'
IMAGE_UPLOADS ="/"

STATIC_DIR = os.path.abspath('static')
application = Flask(__name__, static_folder=STATIC_DIR)
# Enable CORS
CORS(application)
application.config["IMAGE_UPLOADS"] ='data/pt/'
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
     #np.save('train_dara.npy',training_data)"""
	shuffle(training_data)
	return training_data

@application.route("/predict", methods=["POST"])
def predict():
	if request.method == "POST":
		picinput = request.files['img']##รับรูปจากฟอร์มในหน้าเว็ป recive img from <html>
		
		picinput.save(os.path.join(application.config["IMAGE_UPLOADS"],img))
		#id = 'longgong' # Get user id เข้ามาเก็บไว้ตรงนี้ จะได้ตั้งชื่อรูปแบบไม่ซ้ำกันได้
		'''--------------ส่งรูปไปลบ BG------------
		response = requests.post(
    	'https://api.remove.bg/v1.0/removebg',
    	files={'image_file':picinput}, #สังเกตุที่ data เป็นการดึงมาจากฟอร์มใส่ในนี้เลย ไม่ใส่ ' ' ครอบไว้
    	data={'size': 'auto'},
    	headers={'X-Api-Key': 'RGhvvuPjKwbnceGfBPiJsrum'},
    	)
		if response.status_code == requests.codes.ok:
			with open(id+'.png', 'wb') as out: #จากตรงนี้คืรูปที่ออกมา
					toai = out.write(response.content)
					return redirect('/')
		'''
		
		#--------------------
		
		train_data = create_train_data()
		train = train_data[:1]
		
		X_train = np.array([i[0] for i in train]).reshape(-1,100,100,1)
		Y_train = np.array([i[1] for i in train])
		#plt.imshow(X_train[1].reshape(IMG_SIZE,IMG_SIZE),cmap='gist_gray')
		
		load_naja =  load_model(modelpath)
		predicted = load_naja.predict(X_train)
		predicted 
		predicteds =np.argmax(predicted)
		print(data[predicteds])
		
		return data[predicteds]
		#return redirect("/loaded")
		

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.run()
