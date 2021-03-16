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

from cs329s_waymo_object_detection.utils.train_utils import get_custom_backbone_fast_rcnn
from cs329s_waymo_object_detection.utils.gcp_utils import download_blob, upload_blob 

from google.cloud import storage


app = flask.Flask(__name__)
model = None

def load_model(model_weights):
    global model
    model = get_custom_backbone_fast_rcnn(4)
    model.load_state_dict(torch.load('/Users/peterfagan/Downloads/weights_1.pth', map_location=torch.device('cpu')))  # /home/data/waymo/{}'.format(model_weights)
    model.eval()


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            img = flask.request.files["image"]
            img = Image.open(img)
            img = np.array(img)
            img = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(device)
            outputs = model(img)
            
            # Apply NMS here
            keeps = torchvision.ops.nms(outputs[0]['boxes'],outputs[0]['scores'],0.1).to('cpu').numpy()

            boxes = [list(x.detach().numpy().astype(str)) for idx, x in enumerate(outputs[0]['boxes']) if idx in keeps]
            labels = [str(int(x)) for idx, x in enumerate(outputs[0]['labels']) if idx in keeps]
            scores = [str(float(x)) for idx, x in enumerate(outputs[0]['scores']) if idx in keeps]

            annotations = {"boxes":boxes, "labels":labels, "scores":scores}
            predictions = flask.jsonify(annotations)

            data["success"] = True

        if flask.request.files.getlist('images[]'):
            processed_imgs = []
            imgs = flask.request.files.getlist('images[]')
            for img in imgs:
                img = Image.open(img)
                img = np.array(img)
                img = torch.tensor(img).permute(2,0,1).float()
                processed_imgs.append(img)
            
            outputs = model(processed_imgs)
            
            predictions = {}
            cameras = ['FRONT_LEFT','FRONT','FRONT_RIGHT','SIDE_LEFT','SIDE_RIGHT']
            for idx, camera in enumerate(cameras):
                keeps = torchvision.ops.nms(outputs[idx]['boxes'],outputs[idx]['scores'],0.1).to('cpu').numpy()

                boxes = [list(x.detach().numpy().astype(str)) for idx, x in enumerate(outputs[idx]['boxes']) if idx in keeps]
                labels = [str(int(x)) for idx, x in enumerate(outputs[idx]['labels']) if idx in keeps]
                scores = [str(float(x)) for idx, x in enumerate(outputs[idx]['scores']) if idx in keeps]

                annotations = {"boxes":boxes, "labels":labels, "scores":scores}
                predictions[camera] = annotations
            
            predictions = flask.jsonify(predictions)
            data["success"] = True


    return predictions


if __name__ == "__main__":
    print(("* Loading Pytorch model and Flask starting server..."
        "please wait until server has fully started"))
    load_model('weights_13.pth') #hard-code for now
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.run(host='0.0.0.0', debug=False)