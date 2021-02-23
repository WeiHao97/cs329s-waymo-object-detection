import sys
import os

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import wandb


def collate_fn(batch):
    return tuple(zip(*batch))

def get_fast_rcnn(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_custom_backbone_fast_rcnn(num_classes):
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=4)
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model





def track_metrics(loss, classifier_loss, box_reg_loss, objectness_loss, rpn_loss, epoch):
            print('\n')
            print('################################################')
            print('Epoch_{}'.format(epoch))
            print('################################################')
            print('loss: {}'.format(loss))
            print('classifier_loss: {}'.format(classifier_loss))
            print('box_reg_loss: {}'.format(box_reg_loss))
            print('objectness_loss: {}'.format(objectness_loss))
            print('rpn_loss: {}'.format(rpn_loss))
            print('\n')

            wandb.log({'loss':loss})
            wandb.log({'classifier_loss':classifier_loss})
            wandb.log({'box_reg_loss':box_reg_loss})
            wandb.log({'objectness_loss':objectness_loss})
            wandb.log({'rpn_loss':rpn_loss})