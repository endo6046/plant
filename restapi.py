from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array

from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
from keras.models import load_model
import flask
from keras.preprocessing import image
from flask import Flask
from flask_restful import Api

app = Flask(__name__)
api = Api(app)

# initialize our Flask application and the Keras model
model = None


def load():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = load_model('plant.h5')
    model._make_predict_function()




Classes = ["Potato___Early_blight","Potato___Late_blight","Potato___healthy","Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight","Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot","Tomato___Tomato_mosaic_virus","Tomato___healthy"]


def prepare(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = x/255
    img = np.expand_dims(x, axis=0)
    return img


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("img_path"):
            # read the image in PIL format
            image = flask.request.files["img_path"]

            # preprocess the image and prepare it for classification
            image = prepare(image)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict_classes(image)
            result = (Classes[int(preds)])

            # indicate that the request was a success
            data = result

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load()
    app.run()



