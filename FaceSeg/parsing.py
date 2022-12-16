import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2 as cv

from PIL import Image
import PIL

import argparse
import os

from networks import get_model

def vis_parsing_maps(im, parsing_anno, color=[230, 50, 20]):
    # Colors for all 20 parts
    part_colors = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
                   [255, 204, 204], [102, 51, 0], [
                       255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153],
                   [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno_color = np.zeros(
        (parsing_anno.shape[0], parsing_anno.shape[1], 3))

    for pi in range(len(part_colors)):
        index = np.where(parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv.addWeighted(cv.cvtColor(
        vis_im, cv.COLOR_RGB2BGR), 1.0, vis_parsing_anno_color, 0.5, 0)
    vis_im = Image.fromarray(cv.cvtColor(vis_im, cv.COLOR_BGR2RGB))

    return vis_im

net = get_model("EHANet18", False).cuda()
net.load_state_dict(torch.load("./models/EHANet18/39_0.7656_G.pth"))