import os
import json

import numpy as np
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import argparse

import tensorflow as tf

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from google.cloud import storage
from herbie_vision.utils.gcp_utils import download_blob, upload_blob

# Read in script arguments
parser = argparse.ArgumentParser(description='Convert waymo dataset to coco data format in GCP.')
parser.add_argument('path_to_config', type=str,
                    help='path to configuration file')
args = parser.parse_args()


# Read in config file
with open(args['path_to_config']) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =  config['gcp_credentials']

# Task: create config file to store these options
bucket_name = config['gcp_waymo_raw_data_bucket']
processed_bucket = config['gcp_waymo_processed_data_bucket']
root_directory = config['root_directory']
temp_directory = config['temp_directory']
training_folder = config['waymo_training_path']


# Task wrap this logic in a function
# Connect to gcp storage
storage_client = storage.Client()
bucket = storage_client.get_bucket(bucket_name)
blobs = bucket.list_blobs(prefix = training_folder, delimiter='/')

print("Starting processing...")
for blob in tqdm(list(blobs)):
    # download segment
    filename = blob.name.replace(training_folder,"")
    blob.download_to_filename(filename)
    dataset = tf.data.TFRecordDataset(filename, compression_type='')

    # define annotations for a given segment
    annotations = {
            "info":{"description":"Waymo Open Data - {}".format(filename)},
            "licenses":{},
            "images":[],
            "annotations":[],
            "categories":[
                    {
                        "id": 1,
                        "name": "TYPE_VEHICLE"
                    },
                    {
                        "id": 2,
                        "name": "TYPE_PEDESTRIAN"
                    },
                    {
                        "id": 4,
                        "name": "TYPE_CYCLIST"
                    }
                ]
            }

    # process images and annotations
    for idx, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # Get image data
        segment_name = frame.context.name
        segment_metadata = frame.context.stats
        timestamp = frame.timestamp_micros/1000000
        dt_object = datetime.fromtimestamp(timestamp)
        date = dt_object.strftime("%Y-%m-%d")
        
        # Write images to file locations partitioned by day
        for camera_id, image in enumerate(frame.images):
            camera = open_dataset.CameraName.Name.Name(camera_id+1) # Convert to camera name
            img_array = np.array(tf.image.decode_jpeg(image.image))
            img = Image.fromarray(img_array)
            img.save(temp_directory+"{}_{}.jpeg".format(segment_name,camera))
            upload_blob("waymo-processed",temp_directory+"{}_{}.jpeg".format(segment_name, camera),"train/images/{}/{}/{}_{}_{}.jpeg".format(date,segment_name, segment_name, idx, camera))
            os.remove(temp_directory+"{}_{}.jpeg".format(segment_name,camera))
            annotations["images"].append({"id":"{}_{}_{}".format(segment_name, idx, camera), 
                                          "gcp_url":"gs://waymo-processed/train/images/{}/{}/{}_{}_{}.jpeg".format(date,segment_name, segment_name, idx, camera),
                                          "file_name":"{}_{}_{}.jpeg".format(segment_name, idx, camera)})

            for camera_labels in frame.camera_labels:
                if camera_labels.name != image.name:                # Ignore camera labels that do not correspond to this camera.
                    continue
                else:                     # Iterate over the individual labels.
                    for label in camera_labels.labels:
                        bbox = [label.box.center_x - (0.5*label.box.length), 
                                label.box.center_y - (0.5*label.box.width), 
                                label.box.width, label.box.length]
                        annotations["annotations"].append({"id":label.id, 
                                                   "category_id":label.type,
                                                   "image_id":"{}_{}_{}".format(segment_name, idx, camera),
                                                   "area":label.box.length*label.box.width,
                                                   "bbox":bbox})

    with open(temp_directory+"{}.json".format(segment_name), "w") as f:
        json.dump(annotations,f)
    upload_blob("waymo-processed",temp_directory+"{}.json".format(segment_name),"train/annotations/{}/{}.json".format(date,segment_name))
    os.remove(temp_directory+"{}.json".format(segment_name))
    os.remove(root_directory+filename)