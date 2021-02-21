import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import os
import json
from tqdm import tqdm

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from google.cloud import storage
from herbie_vision.utils.gcp_utils import download_blob ,upload_blob


# Task: create config file to store these options
bucket_name = "waymo-raw-data"
processed_bucket = "waymo-processed"
root_directory = "/home/waymo/"
temp_directory = "/home/waymo/temp/"
training_folder = "waymo_open_dataset_v_1_2_0_individual_files/training/"


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