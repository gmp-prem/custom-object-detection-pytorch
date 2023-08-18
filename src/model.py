# this file, we are going to use the pretrained model provided from the Torch framework itself
#  The architecture of object detection framework we use is TWO-STAGE approach (mostly depends on algo)

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes):

    # load Faster RCNN pretrained model
    # in this pytorch version is different from the code provided from the tutorial
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # define new input features and num_classes for the HEAD of faster rcnn
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model