import sys
import os

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import wandb

import json
from herbie_vision.utils.gcp_utils import download_blob, upload_blob
import os


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



def concatenateJSON(paths, root_dir, dataset_type, new_file_name, gcp_bucket="waymo-processed"):
    """
    :param paths: list of annotation file paths to concatenate
    :return: gcp path containing JSON concatenatation of input annotation files

    *** assuming bucket is waymo-processed at this point ***

    *** sample use of function

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/Users/tiffanyshi/Desktop/waymo-2d-object-detection-9ea7bd3b9e0b.json'
    root_dir = '/Users/tiffanyshi/PycharmProjects/329swaymoproject/herbie-vision/'
    tests = ['train/annotations/2019-05-22/11940460932056521663_1760_000_1780_000.json',
             'test/annotations/2019-06-01/2942662230423855469_880_000_900_000.json']
    concatenateJSON(tests, root_dir, "test", "tester.json")

    """

    gcp_curated_annotations_path = "train/curated_annotations/" + new_file_name

    return_file = "tmpFile.json"
    return_dict = {}
    for gcp_annotations_path in paths:

        download_blob(gcp_bucket,
                      gcp_annotations_path,
                      root_dir + dataset_type + '/' + 'annotations.json')

        f = open(root_dir + dataset_type + '/' + 'annotations.json', 'r')
        data = json.load(f)

        if len(return_dict) == 0:
            for key in data.keys():
                if isinstance(data[key], list):
                    return_dict[key] = data[key].copy()
                else:
                    return_dict[key] = list([data[key]])
            f.close()
            continue
        for key in data.keys():
            if isinstance(data[key], list):
                return_dict[key].extend(data[key])
            else:
                return_dict[key].extend(list([data[key]]))
        f.close()

    with open(return_file, "w") as f:
        json.dump(return_dict, f)

    upload_blob(gcp_bucket, return_file, gcp_curated_annotations_path)

    os.remove(return_file)
    os.remove(root_dir + dataset_type + '/' + 'annotations.json')

    return gcp_curated_annotations_path