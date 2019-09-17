import numpy as np
import torch

input = np.array([[100, 100, 54, 63]])


def make_target(BBoxes, image_size):
    grid_num = 7
    boxes = BBoxes
    resize = [image_size[0], image_size[1], image_size[0], image_size[1]]
    boxes = boxes / resize
    target = torch.zeros((grid_num, grid_num, 10))
    cell_size = 1. / grid_num
    wh = torch.from_numpy(boxes[:, 2:])
    cxcy = torch.from_numpy((boxes[:, 2:] + boxes[:, :2]) / 2)
    for i in range(cxcy.shape[0]):
        cxcy_sample = cxcy[i]
        ij = np.ceil(cxcy_sample / cell_size) - 1
        target[int(ij[1]), int(ij[0]), 4] = 1
        target[int(ij[1]), int(ij[0]), 9] = 1
        xy = ij * cell_size
        delta_xy = (cxcy_sample - xy) / cell_size
        target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
        target[int(ij[1]), int(ij[0]), :2] = delta_xy
        target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
        target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
    return target


target = make_target(input, (480, 270))
