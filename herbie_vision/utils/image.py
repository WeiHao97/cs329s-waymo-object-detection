import os
import numpy as np
import cv2
import random
from PIL import Image
import pandas as pd
import glob



def annotations_to_df(annotations, path):
    df = pd.DataFrame(annotations['annotations'])
    df['filename'] = df['image_id'].apply(lambda x :path+'{}.jpeg'.format(x))
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

def create_mask(bb, x):
    """Creates a mask for the bounding box of same shape as image"""
    rows,cols,*_ = x.shape
    Y = np.zeros((rows, cols))
    bb = bb.astype(int)
    Y[bb[1]:bb[3], bb[0]:bb[2]] = 1.
    return Y


def mask_to_bb(Y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(Y)
    if len(cols)==0: 
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([top_row, left_col, bottom_row, right_col], dtype=np.float32)


def resize_image_bb(read_path,write_path,bb,sz):
    """Resize an image and its bounding box and write image to new path"""
    im = np.array(Image.open(read_path))
    im_resized = cv2.resize(im, (sz[0], sz[1]))
    Y_resized = cv2.resize(create_mask(bb, im), (sz[0], sz[1]))
    new_path = write_path + read_path.split('/')[-1]
    cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    return new_path, mask_to_bb(Y_resized)


def process_resizing(resized_path, annotations_df, sz):
    new_paths = []
    new_bbs = []
    for index, row in annotations_df[['filename','x_min','y_min','x_max','y_max']].iterrows():
        new_path,new_bb = resize_image_bb(row['filename'], resized_path,
                                          np.array(row[['x_min','y_min','x_max','y_max']]),sz)
        new_paths.append(new_path)
        new_bbs.append(new_bb)
    annotations_df['processed_filepath'] = new_paths
    annotations_df['resized_bb'] = new_bbs
    annotations_df['xr_min'] = annotations_df['resized_bb'].apply(lambda x: x[0])
    annotations_df['yr_min'] = annotations_df['resized_bb'].apply(lambda x: x[1])
    annotations_df['xr_max'] = annotations_df['resized_bb'].apply(lambda x: x[2])
    annotations_df['yr_max'] = annotations_df['resized_bb'].apply(lambda x: x[3])
    annotations_df.drop(columns='resized_bb',inplace=True)
    
    
    return annotations_df    


def plot_annotations(img, bbox, labels, scores):    
    # Create figure and axes
    fig, ax = plt.subplots(figsize = (200,20))
    
    # Display the image
    ax.imshow(img)
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

    plt.show()
    
    
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