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

from google.cloud import storage
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./creds.json"

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None


def get_custom_backbone_fast_rcnn(num_classes):
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=4)
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )


def load_model():
    global model
    model = get_custom_backbone_fast_rcnn(4)
    client = storage.Client()
    bucket = client.get_bucket('herbie_trained_models') 
    download_blob('herbie_trained_models', 'fastrcnn.pth', './fastrcnn.pth')

    model.load_state_dict(torch.load("./fastrcnn.pth", map_location=torch.device('cpu')))
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
    load_model()
    app.run(host='0.0.0.0', debug=False)