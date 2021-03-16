import os
import numpy as np
import cv2
import random
from PIL import Image
import pandas as pd
import glob

from matplotlib import pyplot as plt
import matplotlib.patches as patches



def annotations_to_df(annotations, mount_dir, image_map):
    df = pd.DataFrame(annotations['annotations'])
    df['gcp_path'] = df['image_id'].apply(lambda x: mount_dir + image_map[x])
    df['x_min'] = df['bbox'].apply(lambda x: x[0])
    df['y_min'] = df['bbox'].apply(lambda x: x[1])
    df['width'] = df['bbox'].apply(lambda x: x[2])
    df['height'] = df['bbox'].apply(lambda x: x[3])
    df['x_max'] = df['x_min'] + df['height']
    df['y_max'] = df['y_min'] + df['width']
    df.drop(columns='bbox',inplace=True)

    return df


def df_to_annotations():
    print('Not implemented yet')


def plot_annotations(img, bbox, labels, scores, confidence_threshold, 
                    save_fig_path='predicted_img.jpeg', show=False, save_fig=True):
    """
    This function plots bounding boxes over image with text labels and saves the image to a particualr location.
    """
    
    # Default colors and mappings
    colors_map={'1':'#5E81AC','2':'#A3BE8C','3':'#B48EAD'}
    labels_map={'1':'Vehicle','2':'Person','3':'Cyclist'}    
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize = (200,200))
    
    # Display the image
    ax.imshow(img)
    i=0
    scores_ind = [idx for idx,x in enumerate(scores) if x>confidence_threshold] # Filter for scores greater than certain threshold
    for idx, entry in enumerate(bbox):
        if idx in scores_ind:
            h = entry[2]-entry[0]
            w = entry[3]-entry[1]
            
            # Create a Rectangle patch
            rect = patches.Rectangle((entry[0],entry[1]), h, w, 
                                    linewidth=60, 
                                    edgecolor=colors_map[str(labels[idx])],
                                    facecolor='none')

            # Add classification category
            plt.text(entry[0], entry[1], s=labels_map[str(labels[idx])], 
                  color='white', verticalalignment='top',
                  bbox={'color': colors_map[str(labels[idx])], 'pad': 0},
                  font={'size':500})

            # Add the patch to the Axes
            ax.add_patch(rect)
        i+=1

    if show==True:
        plt.show()

    plt.savefig(save_fig_path, 
                    bbox_inches = 'tight',
                    pad_inches = 0,
                    dpi=5)

    return save_fig_path
    
    
def write_annotations(model, images_path, write_path, nms_thresh, score_thresh):
    # Hard coded for now improve in future
    colors_map={'1':'#5E81AC','2':'#A3BE8C','3':'#B48EAD'}
    labels_map={'1':'Vehicle','2':'Person','3':'Cyclist'}
     # Read images in folder for given segment camera
    image_files =[]
    for filename in glob.glob(images_path):
        image_files.append(filename)
    image_files.sort(key = lambda x: int(x.split('_')[-2]))
    
    for filename in image_files:
        img = cv2.imread(filename)
        img = torch.tensor(img).permute(2,0,1).float().to(device)
        img = [img]

        model.eval()
        outputs = model(img)

        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox = outputs[0]['boxes']
        scores = outputs[0]['scores']
        labels = outputs[0]['labels']

        keep = torchvision.ops.nms(bbox,scores,nms_thresh)
        labels = [int(x.detach().to('cpu')) for idx, x in enumerate(labels) if idx in keep]
        bbox = [x.detach().to('cpu') for idx, x in enumerate(bbox) if idx in keep]
        scores = [x.detach().to('cpu') for idx, x in enumerate(scores) if idx in keep]
        
        
        # Create figure and axes
        my_dpi=100
        fig, ax = plt.subplots(figsize=(20,10), dpi=my_dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # Display the image
        ax.imshow(img)
        i=0
        scores_ind = [idx for idx,x in enumerate(scores) if x>score_thresh] # Filter for scores greater than certain threshold
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
        plt.savefig(write_path+'{}'.format(filename.split('/')[-1]), 
                    bbox_inches = 'tight',
                    pad_inches = 0,
                    dpi=my_dpi)


def write_video_file(image_path, write_path):
    # Read images in folder for given segment camera
    image_files =[]
    for filename in glob.glob(images_path):
        image_files.append(filename)

    image_files.sort(key = lambda x: int(x.split('_')[-2]))
    
    # Output mp4 video file
    frameSize = (800, 800)
    out = cv2.VideoWriter('/Users/peterfagan/Desktop/test.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 20, frameSize)
    for filename in image_files:
        img = cv2.imread(filename)
        out.write(img)
    out.release()