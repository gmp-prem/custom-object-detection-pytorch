# architecture of YOLO, contains the network that creates the YOLO network

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def parse_cfg(cfgfile):
    """
    this function will take the config file

    ret
    list of blocks. each block describe a block in the NN to be built
    block is represented as a dictionary in the list
    """
    file = open(cfgfile, "r") # store open file from the parsed cfg file
    lines = file.read().split("\n") # split the read file and store in the splitted in the list
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != "#"]
    lines = [x.rstrip().lstrip() for x in lines]
    # print(lines)

if __name__ == "__main__":
    parse_cfg("cfg/yolov3.cfg")