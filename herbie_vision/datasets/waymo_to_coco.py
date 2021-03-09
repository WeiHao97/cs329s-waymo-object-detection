import os
import json

import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import yaml
import argparse
import matplotlib.pyplot as plt


import tensorflow as tf

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from google.cloud import storage
from herbie_vision.utils.gcp_utils import download_blob, upload_blob


def initialize_annotations_dict():
    annotations = {
                "info":{"description":"Waymo Open Data"},
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

    return annotations


def get_metadata(frame):
    segment_metadata = frame.context.stats
    segment_name = frame.context.name
    timestamp = frame.timestamp_micros/1000000
    dt_object = datetime.fromtimestamp(timestamp)
    date = dt_object.strftime("%Y-%m-%d")
    time_of_day = segment_metadata.time_of_day
    location = segment_metadata.location
    weather = segment_metadata.weather
    gcp_url = 'gs://{}/{}/annotations/{}/{}.json'.format(processed_bucket, datatype, date, segment_name)

    return [segment_name, date, time_of_day, location, weather, gcp_url]


def process_segment(dataset, annotations, processed_bucket, temp_directory, root_directory, datatype):
    for idx, data in enumerate(dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            # get segment name and date for constructing filepaths
            segment_name = frame.context.name
            timestamp = frame.timestamp_micros/1000000
            dt_object = datetime.fromtimestamp(timestamp)
            date = dt_object.strftime("%Y-%m-%d")
            # Get segment metadata
            if idx == 1:
                seg_metadata = get_metadata(frame)
                
                    
            # Write images to file locations partitioned by day
            for camera_id, image in enumerate(frame.images):
                camera = open_dataset.CameraName.Name.Name(camera_id+1) # Convert to camera name
                img_array = np.array(tf.image.decode_jpeg(image.image))
                img = Image.fromarray(img_array)
                img.save(temp_directory+"{}_{}.jpeg".format(segment_name,camera))
                upload_blob(processed_bucket,temp_directory+"{}_{}.jpeg".format(segment_name, camera),"{}/images/{}/{}/{}_{}_{}.jpeg".format(datatype, date,segment_name, segment_name, idx, camera))
                os.remove(temp_directory+"{}_{}.jpeg".format(segment_name,camera))
                annotations["images"].append({"id":"{}_{}_{}".format(segment_name, idx, camera), 
                                              "gcp_url":"gs://{}/{}/images/{}/{}/{}_{}_{}.jpeg".format(processed_bucket, datatype, date,segment_name, segment_name, idx, camera), # change this line
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
    upload_blob(processed_bucket,temp_directory+"{}.json".format(segment_name),"{}/annotations/{}/{}.json".format(datatype, date, segment_name))
    os.remove(temp_directory+"{}.json".format(segment_name))

    return seg_metadata


def run_data_processing(bucket_name, processed_bucket, root_directory, temp_directory, folder, datatype):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix = folder, delimiter='/')
    metadata = []

    print("Starting processing...")
    for blob in tqdm(list(blobs)):
        # Ignore the folder specific blob
        if blob.name == folder:
            continue
        else:
            # download segment
            filename = blob.name.replace(folder,"")
            blob.download_to_filename(filename)
            dataset = tf.data.TFRecordDataset(filename, compression_type='')

            # initialize annnotations dictionary
            annotations = initialize_annotations_dict()

            # process images and annotations in frame and return metadata
            temp_metadata = process_segment(dataset, annotations, processed_bucket, temp_directory, root_directory, datatype)
            os.remove(root_directory+filename)
            metadata.append(temp_metadata)
            metadata_df  = pd.DataFrame(metadata, columns = ["segment_name", "date", "time_of_day", "location","weather","gcp_url"])
            metadata_df.to_csv(temp_directory+"metadata.csv",index=False)
            upload_blob(processed_bucket, temp_directory+"metadata.csv","{}/metadata/metadata.csv".format(datatype))


    

if __name__=='__main__':
    # Read in script arguments
    parser = argparse.ArgumentParser(description='Convert waymo dataset to coco data format in GCP.')
    parser.add_argument('path_to_creds_config', type=str,
                        help='path to configuration file')
    parser.add_argument('path_to_dataprocessing_config', type=str,
                        help='path to configuration file')
    args = parser.parse_args()


    # Read in config file
    with open(args.path_to_creds_config) as file:
        base_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.path_to_dataprocessing_config) as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)

    # Setting GCP credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =  base_config['gcp_credentials']

    # Reading in configuration files
    bucket_name = data_config['gcp_waymo_raw_data_bucket']
    processed_bucket = data_config['gcp_waymo_processed_data_bucket']
    root_directory = data_config['root_directory']
    temp_directory = data_config['temp_directory']
    folder = data_config['raw_data_path']
    datatype = data_config['datatype'] 

    # Run dataprocessing steps
    run_data_processing(bucket_name, processed_bucket, root_directory, temp_directory, folder, datatype)