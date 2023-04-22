import numpy as np
import json
import jsonpickle
from flask import Flask, request, Response
from waitress import serve
import cv2

app = Flask(__name__)
model = load_model("guitar_learner.h5")
print("flag")

@app.route("/")
def home():
        return "<h1>chord prediction server</h1>"

@app.route('/api/test', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    save_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    pred_probab, pred_class = keras_predict(model, save_img)
    response = {'message': pred_class}
    print("CHORD IS: ", pred_class)
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")

def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    img = cv2.resize(img, (200, 200))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, 200, 200, 1))
    return img

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=3349, threads=2)