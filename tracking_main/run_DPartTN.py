import numpy as np
import os
import pickle
import sys
import cv2

reload(sys)
sys.setdefaultencoding('utf8')
import time
import argparse
import json
from PIL import Image
import matplotlib.pylab as plt
import torchvision.transforms as transforms

import torch
import torch.utils.data as data
import torch.nn as nn



sys.path.insert(0, '../modules')
from gen_config import *
from DPartTN_Tracker import *

import matplotlib.patches as patches

np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)


def run_mdnet(seqs, rp, SAVE_Image, display=False):
    # Init bbox
    gt = np.asarray(seqs.gtRect)
    img_list = seqs.s_frames
    target_bbox = np.array(gt[:,1])
    result = np.zeros((len(img_list), 4))
    result_bb = np.zeros((len(img_list), 4))
    result[0] = target_bbox
    result_bb[0] = target_bbox

    if gt is not None:
        overlap = np.zeros(len(img_list))
        overlap[0] = 1




    tic = time.time()

    # Load first image
    image = Image.open(img_list[0]).convert('RGB')

    tracker = DPartTN_tracker(image, target_bbox)

    spf_total = time.time() - tic

    # Display
    '''savefig = savefig_dir != ''
    if display or savefig:
        dpi = 80.0
        figsize = (image.size[0] / dpi, image.size[1] / dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image)

        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3],
                                    linewidth=3, edgecolor="b", zorder=1, fill=False)
            ax.add_patch(gt_rect)

        rect = plt.Rectangle(tuple(target_bbox[:2]), target_bbox[2], target_bbox[3],
                              linewidth=3, edgecolor="r", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir, '0000.jpg'), dpi=dpi)'''



    # Main loop
    for i in range(1, len(img_list)):
        tic = time.time()

        image = Image.open(img_list[i]).convert('RGB')

        cover_box = tracker.track(image)

        result_bb[i] = cover_box

        spf = time.time() - tic
        spf_total += spf

        print "Frame %d/%d, Overlap %.3f, Time %.3f" % \
                (i, len(img_list), overlap_ratio(gt[i], cover_box)[0], spf)

    fps = len(img_list) / spf_total
    print(fps)


    return result_bb
