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
from herbie_vision.utils.train_utils import get_fast_rcnn, track_metrics, collate_fn, get_custom_backbone_fast_rcnn, get_map

import sklearn.metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

import wandb


def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
      T = {}
      P = {}
      for imgs, annotations in tqdm(valid_dataloader):
          imgs = [img.to(device) for img in imgs]
          annotations = [{k: v for k, v in t.items()} for t in annotations]
          model_preds = model(imgs)
          for idx, img in enumerate(imgs):
              model_pred = model_preds[idx]
              model_pred = {entry:model_pred[entry].detach().to('cpu').numpy() for entry in model_pred}
              img_annotations = annotations[idx]
              predictions = []
              for pred_idx in range(len(model_pred['boxes'])):
                  pred_box_dict = {}
                  pred_box_dict['class'] = model_pred['labels'][pred_idx]
                  pred_box_dict['x1'] = model_pred['boxes'][pred_idx][0]
                  pred_box_dict['x2'] = model_pred['boxes'][pred_idx][2]
                  pred_box_dict['y1'] = model_pred['boxes'][pred_idx][1]
                  pred_box_dict['y2'] = model_pred['boxes'][pred_idx][3]
                  pred_box_dict['prob'] = model_pred['scores'][pred_idx]
                  predictions.append(pred_box_dict)
              img_data = []
              for ann_idx in range(len(img_annotations['boxes'])):
                  gt_box_dict = {}
                  gt_box_dict['class'] = img_annotations['labels'][ann_idx]
                  gt_box_dict['x1'] = img_annotations['boxes'][ann_idx][0]
                  gt_box_dict['x2'] = img_annotations['boxes'][ann_idx][2]
                  gt_box_dict['y1'] = img_annotations['boxes'][ann_idx][1]
                  gt_box_dict['y2'] = img_annotations['boxes'][ann_idx][3]
                  gt_box_dict['bbox_matched'] = False
                  img_data.append(gt_box_dict)
              t, p = get_map(predictions, img_data)
              for key in t.keys():
                  if key not in T:
                      T[key] = []
                      P[key] = []
                  T[key].extend(t[key])
                  P[key].extend(p[key])
      all_aps = []
      for key in T.keys():
          ap = average_precision_score(T[key], P[key])
          print('{} AP: {}'.format(key, ap))
          all_aps.append(ap)
      print('mAP = {}'.format(np.mean(np.array(all_aps))))
      wandb.log({'mAP':np.mean(np.array(all_aps))})


def train(model, optimizer, lr_scheduler, train_dataloader, valid_dataloader, train_config, wandb_config):
    print('Starting to train model...')
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(wandb_config.num_epochs):
        model.train()
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
        torch.save(model.state_dict(), train_config['root']+"/model_weights/weights_{}.pth".format(epoch))
        wandb.save(train_config['root']+"/model_weights/weights_{}.pth".format(epoch))

        # Evaluation on validation data
        if (epoch!=0)&(epoch%2==0):
          print('Evaluating model on validation set...')
          evaluate(model, valid_dataloader)




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

    wandb.init(config=hyperparameter_defaults, 
                project=train_config['wandb_project'], 
                entity=train_config['wandb_entity'], 
                name=train_config['wandb_name'])
    wandb_config = wandb.config
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =  base_config['gcp_credentials']


    # Initialize datasets + folders
    train_dataset = WaymoDataset(train_config['root_dir'], train_config['train_dataset'], train_config['category_names'], train_config['category_ids'], train_config['resize'],
                                wandb_config.area_limit)

    train_dataset = WaymoDataset(train_config['bucket'],train_config['train_dataset'],train_config['root'], train_config['train_folder'],
                                )
    train_dataloader = data.DataLoader(train_dataset, batch_size=wandb_config.batch_size, collate_fn=collate_fn)

    valid_dataset = WaymoDataset(train_config['root_dir'], train_config['valid_dataset'], train_config['category_names'], train_config['category_ids'], train_config['resize'],
                                wandb_config.area_limit)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=config['batch_size'], collate_fn=collate_fn)


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
    train(model, optimizer, lr_scheduler, train_dataloader, valid_dataloader, train_config, wandb_config)