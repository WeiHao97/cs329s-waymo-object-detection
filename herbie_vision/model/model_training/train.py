import sys
import os
import yaml
import argparse

import numpy as np
from tqdm import tqdm


import torch
import torch.utils.data as data
import torchvision

from herbie_vision.datasets.waymo import WaymoDataset, collate_fn
from herbie_vision.utils.train_utils import get_fast_rcnn, track_metrics, collate_fn, get_custom_backbone_fast_rcnn

import wandb

from mean_average_precision import MetricBuilder




def get_map(pred, gt, f):
    T = {}
    P = {}
    fx, fy = f

    for bbox in gt:
        bbox['bbox_matched'] = False

    pred_probs = np.array([s['prob'] for s in pred])
    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

    for box_idx in box_idx_sorted_by_prob:
        pred_box = pred[box_idx]
        pred_class = pred_box['class']
        pred_x1 = pred_box['x1']
        pred_x2 = pred_box['x2']
        pred_y1 = pred_box['y1']
        pred_y2 = pred_box['y2']
        pred_prob = pred_box['prob']
        if pred_class not in P:
            P[pred_class] = []
            T[pred_class] = []
        P[pred_class].append(pred_prob)
        found_match = False

        for gt_box in gt:
            gt_class = gt_box['class']
            gt_x1 = gt_box['x1']/fx
            gt_x2 = gt_box['x2']/fx
            gt_y1 = gt_box['y1']/fy
            gt_y2 = gt_box['y2']/fy
            gt_seen = gt_box['bbox_matched']
            if gt_class != pred_class:
                continue
            if gt_seen:
                continue
            iou = bbox_intersection_over_union([pred_x1, pred_y1, pred_x2, pred_y2], [gt_x1, gt_y1, gt_x2, gt_y2])
            #iou = data_generators.iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
            if iou >= 0.5:
                found_match = True
                gt_box['bbox_matched'] = True
                break
            else:
                continue

        T[pred_class].append(int(found_match))

    for gt_box in gt:
        if not gt_box['bbox_matched'] and not gt_box['difficult']:
            if gt_box['class'] not in P:
                P[gt_box['class']] = []
                T[gt_box['class']] = []

            T[gt_box['class']].append(1)
            P[gt_box['class']].append(0)

    return T, P

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def evaluate(model, dataloader):
    model.eval()
    print('Not Implemented Yet')


def train(model, optimizer, lr_scheduler, train_dataloader, valid_dataloader, train_config, wandb_config):
    print('Starting to train model...')
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
<<<<<<< HEAD:herbie_vision/train.py
    for epoch in range(2): #(train_config['num_epochs']):
=======
    for epoch in range(wandb_config.num_epochs):
>>>>>>> 5cb383899699d874e68e5334a2fe628609b6381b:herbie_vision/model/model_training/train.py
        print('Starting Epoch_{}'.format(epoch))
        model.train()
        total_losses = []
        classifier_losses = []
        box_reg_losses = []
        objectness_losses = []
        rpn_losses = []
        for imgs, annotations in tqdm(train_dataloader):
            # Push data to device
            imgs = [img.to(device) for img in imgs]
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            
            # Perform forward pass
            loss_dict = model(imgs, annotations)
            print(loss_dict)
            losses = sum(loss for loss in loss_dict.values())

            # Back propagate errors
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # Track loss
            total_losses.append(sum(loss_dict.values()).detach().to('cpu').numpy())
            classifier_losses.append(loss_dict['loss_classifier'].detach().to('cpu').numpy())
            box_reg_losses.append(loss_dict['loss_box_reg'].detach().to('cpu').numpy())
            objectness_losses.append(loss_dict['loss_objectness'].detach().to('cpu').numpy())
            rpn_losses.append(loss_dict['loss_rpn_box_reg'].detach().to('cpu').numpy())


        lr_scheduler.step()

        # Average mertrics over epoch and track
        loss = np.mean(total_losses)
        classifier_loss = np.mean(classifier_losses)
        box_reg_loss = np.mean(box_reg_losses)
        objectness_loss = np.mean(objectness_losses)
        rpn_loss = np.mean(rpn_losses)
        track_metrics(loss, classifier_loss, box_reg_loss, objectness_loss, rpn_loss, epoch)

        print('Saving model weights...')
        torch.save(model.state_dict(), train_config['root']+"model_weights/weights_{}.pth".format(epoch))
        wandb.save(train_config['root']+"model_weights/weights_{}.pth".format(epoch))

        # Evaluation on validation data
        print('Evaluating model on validation set...')
        # evaluate(model, valid_dataloader)




if __name__=="__main__":
    # Read in script arguments
    parser = argparse.ArgumentParser(description='Convert waymo dataset to coco data format in GCP.')
    parser.add_argument('path_to_base_config', type=str,
                        help='path to configuration file')
    parser.add_argument('path_to_train_config', type=str,
                        help='path to configuration file')
    parser.add_argument('num_epochs', type=int,
                         nargs='?',default=25,
                        help='number of epochs')
    parser.add_argument('learning_rate', type=float,
                        nargs='?',default=0.01,
                        help='learning rate')
    parser.add_argument('momentum', type=float,
                        nargs='?',default = 0.9,
                        help='momentum')
    parser.add_argument('weight_decay', type=float,
                        nargs='?',default = 0.0005,
                        help='weight decay')
    parser.add_argument('batch_size', type=int,
                        default=8,
                        nargs='?',help='batch size')
    parser.add_argument('resize', type=list,
                        nargs='?',default =[1152, 768],
                        help='resized dimension of images')
    parser.add_argument('area', type=int,
                        nargs='?',default=50,
                        help='minimum area of bounding box after resizing which is considered for training')
    args = parser.parse_args()

    # Read configuration files
    with open(args.path_to_base_config) as file:
        base_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.path_to_train_config) as file:
        train_config = yaml.load(file, Loader=yaml.FullLoader)


    # Setup weights and biases and gcp connection
    hyperparameter_defaults = dict(
    num_epochs= 25,
    num_classes=4,
    learning_rate=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    batch_size=8,
    area_limit=5000
    )

    wandb.init(config=hyperparameter_defaults)
    wandb_config = wandb.config
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =  base_config['gcp_credentials']


    # Initialize datasets + folders
    train_dataset = WaymoDataset(train_config['bucket'],train_config['train_dataset'],train_config['root'], train_config['train_folder'],
                                train_config['category_names'], train_config['category_ids'], train_config['resize'],
                                wandb_config.area_limit)
    train_dataloader = data.DataLoader(train_dataset, batch_size=wandb_config.batch_size, collate_fn=collate_fn)

    # Omit these while testing scripts
    # valid_dataset = WaymoDataset('waymo-processed', train_config['valid_dataset'],train_config['root'],
    #                             'valid', train_config['category_names'], train_config['category_ids'])
    # valid_dataloader = data.DataLoader(valid_dataset, batch_size=config['batch_size'], collate_fn=collate_fn)

    # test_dataset = WaymoDataset('waymo-processed', train_config['test_dataset'], train_config['root'], 
    #                             'test', train_config['category_names'], train_config['category_ids'])
    # test_dataloader = data.DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn=collate_fn)


    # Initialize model and optimizer
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_custom_backbone_fast_rcnn(train_config['num_classes'])
    model=model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=wandb_config.learning_rate, momentum=wandb_config.momentum, weight_decay=wandb_config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.95)

    # Train model
    train(model, optimizer, lr_scheduler, train_dataloader, None, train_config, wandb_config)