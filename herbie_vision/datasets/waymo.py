import os
import sys
import json

import numpy as np
import pandas as pd
from PIL import Image
import argparse

import torch
import torch.utils.data as data

from herbie_vision.utils.image import annotations_to_df, process_resizing
from herbie_vision.utils.gcp_utils import download_blob, upload_blob

from google.cloud import storage


def collate_fn(batch):
    return tuple(zip(*batch))

class WaymoDataset(data.Dataset):
    def __init__(self, gcp_bucket, gcp_annotations_path, root_dir, dataset_type, cat_names, cat_ids, resize):
        super(WaymoDataset, self).__init__()
        
        # filepaths
        self.gcp_bucket = gcp_bucket
        self.gcp_annotations_path = gcp_annotations_path
        self.root_dir = root_dir
        self.dataset_type = dataset_type 
        self.local_path_to_images = self.root_dir+self.dataset_type+'/images/'
        self.local_path_to_processed_images = self.root_dir+self.dataset_type+'/images_processed/'
        self.local_path_to_weights = self.root_dir+'model_weights/'
        
        # high level summary values
        self.num_classes = len(cat_names)
        self.category_names = cat_names
        self.category_ids = cat_ids
        self.resize = resize
        
        # setup data directory
        print('Setting up data directories...')
        if os.path.exists(self.root_dir)==False:
            os.mkdir(self.root_dir)
        if os.path.exists(self.local_path_to_weights)==False:
            os.mkdir(self.local_path_to_weights)
        if os.path.exists(self.root_dir+self.dataset_type+'/')==False:
            os.mkdir(self.root_dir+self.dataset_type+'/')
            os.mkdir(self.local_path_to_images)
            os.mkdir(self.local_path_to_processed_images)

            # read in annotations
            client = storage.Client()
            bucket = client.get_bucket(self.gcp_bucket)        
            download_blob(self.gcp_bucket,
                            self.gcp_annotations_path,
                            self.root_dir+ self.dataset_type + '/' + 'annotations.json')
            
            f = open(self.root_dir+ self.dataset_type + '/' + 'annotations.json','r')
            self.annotations = json.load(f)
            f.close()
            

        
            
            print('Downloading and processing images...')
            # convert annotations to dataframe
            self.annotations_df = annotations_to_df(self.annotations, self.local_path_to_images)
            self.annotations_df['category_id'] = self.annotations_df['category_id'].apply(lambda x: 3 if x==4 else x) # map so categorise are contiguous


            
            # determine segment paths
            self.segment_paths = []
            for image in self.annotations['images']:
                uri = image['gcp_url']
                segment = '/'.join(uri.split('/')[3:7])+'/'
                if segment not in self.segment_paths:
                    self.segment_paths.append(segment)
            
            
            # Download images for segments to local folder
            for segment in self.segment_paths:
                blobs = bucket.list_blobs(prefix=segment, delimiter='/')
                for blob in list(blobs):
                    filename=blob.name.replace(segment,'')
                    blob.download_to_filename(self.local_path_to_images+'{}'.format(filename))

            # Preprocess images to be the same size
            print('Processing images...')
            self.annotations_df = process_resizing(self.local_path_to_processed_images, self.annotations_df,resize)
            self.annotations_df.to_csv(self.root_dir+ self.dataset_type + '/processed_annotations.csv' )
        else:
            # read in annotations
            client = storage.Client()
            bucket = client.get_bucket(self.gcp_bucket)        
            download_blob(self.gcp_bucket,
                            self.gcp_annotations_path,
                            self.root_dir+ self.dataset_type + '/' + 'annotations.json')
            
            f = open(self.root_dir+ self.dataset_type + '/' + 'annotations.json','r')
            self.annotations = json.load(f)
            f.close()
            self.annotations_df = pd.read_csv(self.root_dir+ self.dataset_type + '/processed_annotations.csv')

        # Drop bounding boxes which get reduced too much by resizing
        self.annotations_df['r_area'] = (self.annotations_df['xr_max'] - self.annotations_df['xr_min'])*(self.annotations_df['yr_max'] - self.annotations_df['yr_min'])
        self.annotations_df = self.annotations_df[self.annotations_df['r_area']>10]
        

        # Drop images without annotations
        self.annotations['images'] = [x for x in self.annotations['images'] if x['id'] in self.annotations_df['image_id'].unique()]
        
        
        
            
            
    def __getitem__(self, idx):
        image_id = self.annotations['images'][idx]['id']
        image_url = self.annotations['images'][idx]['gcp_url']
        filename = image_url.split('/')[-1]
        image = Image.open(self.local_path_to_processed_images+'{}'.format(filename))
        image = np.asarray(image, dtype="float64") / 255.
        image = torch.tensor(image).permute(2,0,1).float()        
        
        # define target data for fast rcnn
        temp_df = self.annotations_df[self.annotations_df['image_id']==image_id]

        boxes = []
        labels = []
        areas = []
        for _,item in temp_df.iterrows():
            boxes.append([item['xr_min'],item['yr_min'],item['xr_max'],item['yr_max']])
            labels.append(item['category_id'])
            areas.append(item['area'])
        
        boxes = torch.tensor(boxes, dtype=torch.int64)
        areas = torch.tensor(areas, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor(idx)
        target["area"] = areas
        target["iscrowd"] = torch.zeros((temp_df.shape[0],), dtype=torch.int64)
        
        return image, target
    
    def __len__(self):
        return len(self.annotations['images'])