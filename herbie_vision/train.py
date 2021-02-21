import sys
import os

import torch
import torch.utils.data as data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from herbie_vision.datasets.waymo import WaymoDataset, collate_fn
from herbie_vision.utils 

import wandb

wandb.init(project='waymo-2d-object-detection', entity='peterdavidfagan', name='FassRCNNTest')
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =  


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# Basic parameters
NUM_EPOCHS = 10
NUM_CLASSES = 3

# Initialize model and device
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model_instance_segmentation(NUM_CLASSES)
model=model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)


# Initialize dataset
dataset = WaymoDataset('waymo-processed','train/annotations/2019-02-13/10017090168044687777_6380_000_6400_000.json','./data/images/','./data/images_processed/', CATEGORY_NAMES, CATEGORY_IDS)
train_dataloader = data.DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
len_dataloader = len(train_dataloader)


# Try train a number of epochs
for epoch in range(NUM_EPOCHS):
    model.train()
    i = 0    
    for imgs, annotations in train_dataloader:
        i += 1
        imgs = list(img for img in imgs)
        annotations = [{k: v for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')
        wandb.log("Loss":{losses})

    torch.save(model.state_dict(), "./test_{}.pth".format(epoch))
    wandb.save("./test_{}.pth".format(epoch))


