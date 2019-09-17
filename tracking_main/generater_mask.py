import numpy as np







def generater_mask(sample,image):
    size=image.size
    mask = np.zeros((size[0],size[1]))
    for i in range(sample.shape[0]):
        mask_sample=sample[i]
        x1_indx=int(round(mask_sample[0]))
        x2_indx=int(round(mask_sample[0]+mask_sample[2]))
        y1_indx=int(round(mask_sample[1]))
        y2_indx=int(round(mask_sample[1]+mask_sample[3]))
        mask[x1_indx:x2_indx , y1_indx:y2_indx]=1



    return mask


