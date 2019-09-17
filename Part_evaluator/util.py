import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def neg_generate(mask,gt):
    [w, h] = gt[2:4]
    [p_w, p_h] = [w / 7, h / 7]

    start_x = gt[0] - 1.5 * gt[2]
    start_y = gt[1] - 1.5 * gt[3]
    end_x = gt[0] + 2.5 * gt[2]
    end_y = gt[1] + 2.5 * gt[3]

    if (start_x<0): start_x=0
    if (start_y < 0): start_y = 0
    if (end_x+p_w>mask.shape[1]): end_x = mask.shape[1]-p_w
    if (end_y+p_h>mask.shape[0]): end_y = mask.shape[0]-p_h

    partial_box = []

    while (len(partial_box) < 14):
        p_x = np.random.uniform(start_x, end_x)
        p_y = np.random.uniform(start_y, end_y)

        if (mask[round(p_y):round(p_y + p_h), round(p_x):round(p_x + p_w)].sum() == 0):
            partial_box.append([round(p_x), round(p_y), round(p_w), round(p_h)])
    partial_box = partial_box[0:14]
    np.random.shuffle(partial_box)
    return partial_box

def part_generate(mask, gt):

    [w, h] = gt[2:4]

    [p_w,p_h] = [w/7,h/7]
    p_x = np.random.uniform(gt[0],gt[0]+gt[2])
    p_y = np.random.uniform(gt[1],gt[1]+gt[3])

    partial_box = []
    grid_num =6

    while (len(partial_box) < 40):
        grid_num += 1

        if ((grid_num > w) | (grid_num > h)):
            break
        for i in range(grid_num):
            for j in range(grid_num):
                start_x = gt[0] + ((gt[2]) / grid_num) * i
                end_x = start_x + ((gt[2]) / grid_num)
                start_y = gt[1] + ((gt[3]) / grid_num) * j
                end_y = start_y + ((gt[3]) / grid_num)
                p_x = np.random.uniform(start_x, end_x)
                p_y = np.random.uniform(start_y, end_y)
                if (mask[round(p_y):round(p_y + p_h), round(p_x):round(p_x + p_w)].sum() / (p_w * p_h) > 0.5):
                    partial_box.append([round(p_x), round(p_y), round(p_w), round(p_h)])
    np.random.shuffle(partial_box)
    partial_box = partial_box[0:14]


    '''
    for i in range(0,round(w-p_w),round(p_w/4)):
        for j in range(0,round(h-p_h),round(p_h/4)):
            if(mask[round(gt[1]+j):round(gt[1]+j+p_h),round(gt[0]+i):round(gt[0]+i+p_w)].sum()/(p_w*p_h)>0.5):
                partial_box.append([i,j,round(p_w),round(p_h)])
    '''

    '''
    fig, ax = plt.subplots(1)
    ax.imshow(mask)
    rect = patches.Rectangle((gt[0], gt[1]), gt[2], gt[3],
                             linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect)
    for i in range(40):
        rect = patches.Rectangle((partial_box[i][0], partial_box[i][1]), partial_box[i][2], partial_box[i][3],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)


    plt.show()
    '''
    return partial_box