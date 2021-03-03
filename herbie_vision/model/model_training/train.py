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
from herbie_vision.utils.train_utils import get_fast_rcnn, track_metrics, collate_fn

import wandb


def evaluate(model, dataloader):
    model.eval()
    print('Not Implemented Yet')


def train(model, optimizer, lr_scheduler, train_dataloader, valid_dataloader, train_config):
    print('Starting to train model...')
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(train_config['num_epochs']):
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
    args = parser.parse_args()

    # Read configuration files
    with open(args.path_to_base_config) as file:
        base_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.path_to_train_config) as file:
        train_config = yaml.load(file, Loader=yaml.FullLoader)


    # Resolve external dependencies
    wandb.init(project=base_config['project'], entity=base_config['entity'], name=base_config['run_name'])
    
    # Omit while testing on colab
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =  base_config['gcp_credentials']


    # Initialize datasets + folders
    train_dataset = WaymoDataset('waymo-processed',train_config['train_dataset'],train_config['root'], 'train',
                                train_config['category_names'], train_config['category_ids'], train_config['resize'],
                                train_config['area_limit'])
    train_dataloader = data.DataLoader(train_dataset, batch_size=train_config['batch_size'], collate_fn=collate_fn)

    # Omit these while testing scripts
    # valid_dataset = WaymoDataset('waymo-processed', train_config['valid_dataset'],train_config['root'],
    #                             'valid', train_config['category_names'], train_config['category_ids'])
    # valid_dataloader = data.DataLoader(valid_dataset, batch_size=config['batch_size'], collate_fn=collate_fn)

    # test_dataset = WaymoDataset('waymo-processed', train_config['test_dataset'], train_config['root'], 
    #                             'test', train_config['category_names'], train_config['category_ids'])
    # test_dataloader = data.DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn=collate_fn)


    # Training parameters
    NUM_EPOCHS = train_config['num_epochs']
    NUM_CLASSES = train_config['num_classes']
    LR = train_config['learning_rate']
    MOMENTUM = train_config['momentum']
    WEIGHT_DECAY = train_config['weight_decay']


    # Initialize model and optimizer
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_custom_backbone_fast_rcnn(NUM_CLASSES)
    model=model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.95)

    # Train model
    train(model, optimizer, lr_scheduler, train_dataloader, valid_dataloader, train_config)



