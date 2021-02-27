import os
import sys
import requests

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import cv2
# from herbie_vision.utils.train_utils import get_customer_backbone_fast_rcnn
# from herbie_vision.utils.gcp_utils import download_blob, upload_blob

from google.cloud import storage


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



def detect_object(request):
  # Instantiate model
  model = get_custom_backbone_fast_rcnn(4)

  # Download model weights
  client = storage.Client()
  bucket = client.get_bucket('herbie_trained_models') 
  download_blob('herbie_trained_models', 'fastrcnn.pth', '/tmp/fastrcnn.pth')

  # Read in model weights 
  model.load_state_dict(torch.load("/tmp/fastrcnn.pth", map_location=torch.device('cpu')))
  model.eval()

  # Get method
  if request.method == 'GET':
        return "Welcome to Herbie Vision"

  # Accept/Read user request and convert data to tensor
  if request.method == 'POST':

      data = request.get_json()
      img_file = data['image_uri']

      # Read in image file
      filename = img_file.split('/')[-1]
      download_blob('herbie_user_input', filename, '/tmp/{}'.format(filename))
      img = cv2.imread('/tmp/{}'.format(filename))
      img = torch.tensor(img).permute(2,0,1).float()    

      # Perform prediction
      outputs = model(img)
      print(outputs)

      # Create figure
      img = cv2.imread(filename)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      bbox = outputs[0]['boxes']
      scores = outputs[0]['scores']
      labels = outputs[0]['labels']

      keep = torchvision.ops.nms(bbox,scores,0.2)
      labels = [int(x.detach().to('cpu')) for idx, x in enumerate(labels) if idx in keep]
      bbox = [x.detach().to('cpu') for idx, x in enumerate(bbox) if idx in keep]

      my_dpi=100
      fig, ax = plt.subplots(figsize=(20,10), dpi=my_dpi)
      ax = plt.Axes(fig, [0., 0., 1., 1.])
      ax.set_axis_off()
      fig.add_axes(ax)

      i=0
      scores_ind = [idx for idx,x in enumerate(scores) if x>0.4] # Filter for scores greater than certain threshold
      for idx, entry in enumerate(bbox):
          if idx in scores_ind:
              h = entry[2]-entry[0]
              w = entry[3]-entry[1]

              # Create a Rectangle patch
              rect = patches.Rectangle((entry[0],entry[1]), h, w, linewidth=4, edgecolor=colors_map[str(labels[idx])], facecolor='none')

              # Add classification category
              plt.text(entry[0], entry[1], s=labels_map[str(labels[idx])], 
                      color='white', verticalalignment='top',
                      bbox={'color': colors_map[str(labels[idx])], 'pad': 0},
                      font={'size':25})

          # Add the patch to the Axes
          ax.add_patch(rect)
          i+=1
      ax.imshow(img, aspect='auto')
      plt.savefig('/tmp/{}'.format(filename), 
                bbox_inches = 'tight',
                pad_inches = 0,
                dpi=my_dpi)

      upload_blob('herbie_user_input','/tmp/{}'.format(filename),filename)


  return "gs://herbie_user_input/{}".format(filename)