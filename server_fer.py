from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
from flask import Flask, render_template, request
import keras
import cv2 as cv
from keras.models import model_from_json

app = flask.Flask(__name__)
model = None


def load_model():
    global classes_name
    global model
    global face_cascade
    classes_name = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    json_file = open('models/model2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("models/Fer2013.hdf5")

def prepare_image(image, target):

    # img = cv.imread(image)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    x, y, w, h = faces[0]
    gray_face = gray[y:y+h, x:x+w]
    gray_face = cv.resize(gray_face, (48,48))
    test_face =  gray_face[np.newaxis, :, :, np.newaxis]/255.
    return test_face

@app.route('/upload')
def upload():
	return flask.render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		print(request)
		print(request.files)
		f = request.files.get('image')
		f.save(f.filename)
		return 'file uploaded successfully'


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = np.array(Image.open(io.BytesIO(image)))
            image = prepare_image(image, target=(48, 48))
            preds = model.predict(image)
            result = {}
            for i in range(preds.shape[1]):
                result[classes_name[i]] = preds[0][i]
            data["predictions"] = []
            for key in result.keys():
                r = {"label": key, "probability": float(result[key])}
                data["predictions"].append(r)
            data["success"] = True
        return flask.jsonify(data)
if __name__ == "__main__":
	load_model()
	app.run(host="localhost", port=5000,debug = False, threaded = False)
	#app.run(debug = False, threaded = False)	

