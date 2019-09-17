import numpy as np
from PIL import Image
import torch

from utils import *

def NMS_BB_mdnet_reference(bboxes,ref, threshold=0.4,jot=False):
    if jot == True:
        ref = ref[0]
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 0] + bboxes[:, 2]
    y2 = bboxes[:, 1] + bboxes[:, 3]

    rx1 = ref[0]
    ry1 = ref[1]
    rx2 = ref[0] + ref[2]
    ry2 = ref[1] + ref[3]

    order = torch.LongTensor(range(bboxes.shape[0]))
    check = torch.ones_like(order)


    for idx in range(ref.shape[0]):
        i = idx
        xx1 = x1[order].clamp(min=rx1)
        xx2 = x2[order].clamp(max=rx2)
        yy1 = y1[order].clamp(min=ry1)
        yy2 = y2[order].clamp(max=ry2)
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        area = torch.mul(x2 - x1, y2 - y1)
        ref_area = torch.mul(rx2-rx1,ry2-ry1)
        ovr = inter / (ref_area + area - inter)

        ids = order[(ovr > threshold).nonzero()]

        for jdx in range(len(ids)):
            j=ids[jdx].item()
            check[j] = 0

    return check , 1 - check

def neg_generate_gauss(img,gt,ref_part,n,img_size):
    img = np.asarray(img)
    [w, h] = gt[2:4]
    [p_w, p_h] = ref_part[2:]

    p_w = np.clip(p_w, 6, min(img_size) - 10)
    p_h = np.clip(p_h, 6, min(img_size) - 10)

    middle = [gt[0] + gt[2] / 2 - p_w, gt[1] + gt[3] / 2 - p_h]
    new_cand = np.zeros((0,4))

    while new_cand.shape[0] < n:
        cand = np.clip(0.4 * np.random.randn(n, 4), -1, 1) * max(w, h) + [middle[0], middle[1], 0., 0.]
        cand = np.clip(cand, [0, 0, 0, 0], [img.shape[1] - p_w, img.shape[0] - p_h, 0, 0])
        cand[:, 2] = p_w
        cand[:, 3] = p_h

        cand = np.asarray(cand)
        cand = cand.astype('int32')

        check = NMS_BB_mdnet_reference(torch.DoubleTensor(cand),
                                       torch.DoubleTensor([[0, 0, img.shape[1] - p_w, img.shape[0] - p_h]]),
                                       threshold=0.0, jot=True)
        where = np.where(check[0].cpu().numpy())
        where = where[0]
        new_cand_add = np.delete(cand, where, 0)

        if(new_cand_add.shape[0]>0):
            # check = NMS_BB_mdnet_reference(torch.DoubleTensor(new_cand), torch.DoubleTensor([gt]), threshold=0.0)
            check = NMS_BB_mdnet_reference(torch.DoubleTensor(new_cand_add), torch.DoubleTensor(gt), threshold=0.0)
            where = np.where(check[1].cpu().numpy())
            where = where[0]
            new_cand_add = np.delete(new_cand_add, where, 0)

        new_cand = np.append(new_cand, new_cand_add, axis=0)
    new_cand = new_cand[:n,:]
    new_cand[:, 2:] = np.clip(new_cand[:, 2:], 6, min(img_size) - 10)

    return new_cand


def generator(bb,n,img_size):
    # bb: target bbox (min_x,min_y,w,h)
    bb = np.array(bb, dtype='float32')

    samples = np.tile(bb[None, :], (n, 1))

    samples[:, :2] += 0.15 * np.mean(bb[2:]) * np.clip(0.5 * np.random.randn(n, 2), -1, 1)

    samples[:, 2:] = np.clip(samples[:, 2:], 6, min(img_size) - 10)

    return samples



def gen_samples_new(img , bbox, n, cover, numk=False,image_size = 0):
    if numk == True:
        sample = generator(bbox, n,image_size)
    else:
        sample = neg_generate_gauss(img, cover, bbox, n,image_size)

    return sample
