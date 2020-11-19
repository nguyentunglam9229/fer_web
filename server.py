from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
from flask import Flask, render_template, request
import keras

app = flask.Flask(__name__)
model = None

def load_model():
	global model
	
	#global model
	#model = ResNet50(weights="imagenet")
	#model.save("resnet50_model.h5")
	model = keras.models.load_model('models/resnet50_model.h5')	
	#model = ResNet50(weights="imagenet")

def prepare_image(image, target):
	if image.mode != "RGB":
		image = image.convert("RGB")
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	return image

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
			image = Image.open(io.BytesIO(image))

			image = prepare_image(image, target=(224, 224))
			preds = model.predict(image)
			results = imagenet_utils.decode_predictions(preds)
			data["predictions"] = []

			for (imagenetID, label, prob) in results[0]:
				r = {"label": label, "probability": float(prob)}
				data["predictions"].append(r)
			data["success"] = True
	return flask.jsonify(data)
if __name__ == "__main__":
	load_model()
	app.run(host="localhost", port=5000,debug = True, threaded = False)
	#app.run(debug = False, threaded = False)	

