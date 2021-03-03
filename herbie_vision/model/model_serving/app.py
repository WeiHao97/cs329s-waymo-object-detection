import flask
import os
import sys

import numpy as np
import cv2
from PIL import Image

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from herbie_vision.utils.train_utils import get_custom_backbone_fast_rcnn
from herbie_vision.utils.gcp_utils import download_blob, upload_blob 


from google.cloud import storage
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./creds.json"

app = flask.Flask(__name__)
model = None

def load_model(model_weights):
    global model
    model = get_custom_backbone_fast_rcnn(4)
    client = storage.Client()
    bucket = client.get_bucket('herbie_trained_models') 
    download_blob('herbie_trained_models', model_weights, './{}'.format(model_weights))

    model.load_state_dict(torch.load('./{}'.format(model_weights), map_location=torch.device('cpu')))
    model.eval()


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            img = flask.request.files["image"]
            img = Image.open(img)
            img = np.array(img)
            img = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()  
            outputs = model(img)

            boxes = [list(x.detach().numpy().astype(str)) for x in outputs[0]['boxes']]
            labels = [str(int(x)) for x in outputs[0]['labels']]
            scores = [str(float(x)) for x in outputs[0]['scores']]

            annotations = {"boxes":boxes, "labels":labels, "scores":scores}

            data["success"] = True

    return flask.jsonify(annotations)


if __name__ == "__main__":
    print(("* Loading Pytorch model and Flask starting server..."
        "please wait until server has fully started"))
    load_model('fastrcnn.pth') #hard-code for now
    app.run(host='0.0.0.0', debug=False)