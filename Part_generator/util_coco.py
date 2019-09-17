import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def part_generate(mask, gt):

    [w, h] = gt[2:]

    [p_w_ref,p_h_ref] = [w/7,h/7]
    #p_x = np.random.uniform(gt[0],gt[0]+gt[2])
    #p_y = np.random.uniform(gt[1],gt[1]+gt[3])

    partial_box = []
    grid_num =4

    min_size = 20

    max_p_w = (w/3)
    max_p_h = (h/3)

    std = 0.9
    p_w_mean = (min_size+max_p_w)/2
    p_h_mean = (min_size+max_p_h)/2


    while(len(partial_box)<50):
        grid_num+=1
        if((grid_num>w) | (grid_num>h)):
            break
        for i in range(grid_num):
            for j in range(grid_num):
                p_w = np.clip((std * np.random.randn()) * (max_p_w - p_w_mean) / 2 + p_w_mean, min_size, max_p_w)
                p_h = np.clip((std * np.random.randn()) * (max_p_h - p_h_mean) / 2 + p_h_mean, min_size, max_p_w)

                start_x = gt[0]+((gt[2])/grid_num)*i
                end_x = start_x+((gt[2])/grid_num)
                start_y = gt[1]+((gt[3])/grid_num)*j
                end_y = start_y+((gt[3])/grid_num)
                p_x = np.random.uniform(start_x,end_x)
                p_y = np.random.uniform(start_y,end_y)
                if (p_x + p_w > gt[0]+gt[2]):
                    p_w = (gt[0]+gt[2]) - p_x
                if (p_y + p_h > gt[1]+gt[3]):
                    p_h = (gt[1]+gt[3]) - p_y
                if (mask[int(round(p_y)):int(round(p_y+ p_h)), int(round(p_x )):int(round(p_x + p_w))].sum() / (p_w * p_h) > 0.3):
                    partial_box.append([int(round(p_x)), int(round(p_y)), int(round(p_w)), int(round(p_h))])


    partial_box = partial_box

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