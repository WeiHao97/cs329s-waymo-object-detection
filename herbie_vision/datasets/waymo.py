import os
import sys
import json

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import argparse
from multiprocessing import Manager, Pool


import torch
import torch.utils.data as data

from herbie_vision.utils.image import annotations_to_df, process_resizing
from herbie_vision.utils.gcp_utils import download_blob, upload_blob

from google.cloud import storage


def collate_fn(batch):
    return tuple(zip(*batch))

class WaymoDataset(data.Dataset):
    def __init__(self, mount_dir, gcp_annotations_path, cat_names, cat_ids, resize, area_limit, test_dataset=False):
        super(WaymoDataset, self).__init__()
        
        # filepaths
        self.gcp_annotations_path = gcp_annotations_path
        self.mount_dir = mount_dir

        self.dataset_name =  self.gcp_annotations_path.split('/')[-1].replace('.json','')
        self.dataset_path =  self.mount_dir +'/'.join(self.gcp_annotations_path.split('/')[:-2]) 
        self.path_to_annotations = self.mount_dir + self.gcp_annotations_path
        self.path_to_processed_images = self.dataset_path+'/processed_images/'
        
        # high level summary values
        self.num_classes = len(cat_names)
        self.category_names = cat_names
        self.category_ids = cat_ids
        self.resize = resize
        self.area_limit = area_limit
        
        # multiprocessing for image transformations
        manager = Manager()
        self.shared_list = manager.list()
    
        
        # setup data directory
        print('Setting up data directories...')
        if os.path.exists(self.path_to_processed_images)==False:
            if test_dataset==False:
                os.mkdir(self.path_to_processed_images)

                # read annotations file
                f = open(self.path_to_annotations,'r')
                self.annotations = json.load(f)
                f.close()

                # convert annotations to dataframe
                print('Processing images...')
                image_map = {entry['id']:'/'+'/'.join(entry['gcp_url'].split('/')[3::]) for entry in self.annotations['images']}
                self.annotations_df = annotations_to_df(self.annotations, self.mount_dir, image_map)
                self.annotations_df['category_id'] = self.annotations_df['category_id'].apply(lambda x: 3 if x==4 else x) # map so categories are contiguous

                # Resize images to be the same size
                images = [x for x in self.annotations_df.image_id.unique() if int(x.split('_')[5])%5==0] # take every 15th frame from the segment
                pool = Pool()
                pool.map(self.process_image, images)
                pool.close()
                self.shared_list = [item for sublist in self.shared_list for item in sublist]  #flatten
                self.annotations_df = pd.DataFrame(self.shared_list, columns = ['id','category_id','image_id','area','gcp_path',
                                                                                'x_min','y_min','width','height','x_max','y_max'])
                self.annotations_df.to_csv('/'.join(self.path_to_annotations.split('/')[:-1]) + '/processed_annotations.csv', index=False)
            else:
                os.mkdir(self.path_to_processed_images)
                
                # read annotations file
                f = open(self.path_to_annotations,'r')
                self.annotations = json.load(f)
                f.close()
                
                image_map = {entry['id']:'/'+'/'.join(entry['gcp_url'].split('/')[3::]) for entry in self.annotations['images']}

                for entry in self.annotations['images']:
                    img = cv2.imread(self.mount_dir + image_map[entry['id']])
                    img_resized = cv2.resize(img, (self.resize[0], self.resize[1]), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(self.path_to_processed_images+entry['file_name'], img_resized)
        else:
            # read in annotations
            f = open(self.path_to_annotations,'r')
            self.annotations = json.load(f)
            f.close()
            self.annotations_df = pd.read_csv('/'.join(self.path_to_annotations.split('/')[:-1]) + '/processed_annotations.csv')

        # Drop bounding boxes which are too small
        self.annotations_df['area'] = (self.annotations_df['x_max'] - self.annotations_df['x_min'])*(self.annotations_df['y_max'] - self.annotations_df['y_min'])
        self.annotations_df = self.annotations_df[self.annotations_df['area']>self.area_limit]
        

        # Drop images without annotations
        self.annotations['images'] = [x for x in self.annotations['images'] if x['id'] in self.annotations_df['image_id'].unique()]
        self.annotations['images'] = [x for x in self.annotations['images'] if x['id'] in self.annotations_df['image_id'].unique()]
        
        
    def process_image(self, image):
        tmp_df = self.annotations_df[self.annotations_df['image_id']==image]
        img = cv2.imread(tmp_df['gcp_path'].unique()[0])
        img_resized = cv2.resize(img, (self.resize[0], self.resize[1]), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(self.path_to_processed_images+tmp_df['gcp_path'].unique()[0].split('/')[-1], img_resized)
        scale = np.flipud(np.divide(img_resized.shape[:-1], img.shape[:-1]))
        for index, row in tmp_df.iterrows():
            tmp_df.loc[index,['x_min','x_max']] *=scale[0]
            tmp_df.loc[index,['y_min','y_max']] *=scale[1]
            tmp_df.loc[index,'height'] = tmp_df.loc[index,'x_max'] - tmp_df.loc[index,'x_min']
            tmp_df.loc[index,'width'] =  tmp_df.loc[index,'y_max'] - tmp_df.loc[index,'y_min']
        self.shared_list.append(tmp_df.values)
        
            
    def __getitem__(self, idx):
        image_id = self.annotations['images'][idx]['id']
        image_url = self.annotations['images'][idx]['gcp_url']
        filename = image_url.split('/')[-1]
        image = cv2.imread(self.path_to_processed_images+'{}'.format(filename))
        image = torch.tensor(image).permute(2,0,1).float()        
        
        # define target data for fast rcnn
        temp_df = self.annotations_df[self.annotations_df['image_id']==image_id]

        boxes = []
        labels = []
        areas = []
        for _,item in temp_df.iterrows():
            boxes.append([item['x_min'],item['y_min'],item['x_max'],item['y_max']])
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
        
        return image, target
    
    
    def __len__(self):
        return len(self.annotations['images'])