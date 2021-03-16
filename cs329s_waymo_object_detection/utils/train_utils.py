import sys
import os
import numpy as np

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import wandb

import json
from cs329s_waymo_object_detection.utils.gcp_utils import download_blob, upload_blob
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


def classify_record(pred_label, gt_label, iou, iou_thresh):
  if pred_label==gt_label:
    if iou >= iou_thresh:
      return 'TP'
    else:
      return 'FP'
  else:
      if iou >= iou_thresh:
        return 'FN' 
      else:
        return 'TN'


def calc_precision_recall(eval_df, label, iou_thresh):
  try:
    tmp_df = eval_df[eval_df['gt_label']==label]
    tmp_df = tmp_df.sort_values(by='confidence_score', ascending=False).reset_index(drop=True)
    total_positives = tmp_df.shape[0]
    tmp_df['classification'] = tmp_df.apply(lambda x: classify_record(x['pred_label'], x['gt_label'], x['iou'], 0.5), axis=1)

    precision = []
    recall = []
    counts = {'TP':0,'FP':0,'TN':0,'FN':0}
    for classification in list(tmp_df['classification']):
        counts[classification] +=1
        precision.append(counts['TP']/(counts['TP']+counts['FP']))
        recall.append(counts['TP']/total_positives)

    return precision, recall
  except:
    return None, None


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


def concatenateJSON(paths, mount_dir, write_path, gcp_bucket="waymo-processed"):
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

    return_file = "tmpFile.json"
    return_dict = {}
    for gcp_annotations_path in paths:

        f = open(mount_dir + gcp_annotations_path, 'r')
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

    upload_blob(gcp_bucket, return_file, write_path)

    os.remove(return_file)