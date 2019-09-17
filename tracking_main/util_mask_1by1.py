import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_mask(mask_p,mask_image):
    for i in range(mask_image.shape[0]):
        x1_indx = int(round(mask_p[i,0]))
        x2_indx = int(round(mask_p[i,0] + mask_p[i,2]))
        y1_indx = int(round(mask_p[i,1]))
        y2_indx = int(round(mask_p[i,1] + mask_p[i,3]))
        mask_image[i, y1_indx:y2_indx, x1_indx:x2_indx] = 1
    return mask_image

def mask_generate2(image, mask_p):
    mask_all = np.zeros((1, image.shape[0], image.shape[1]), dtype=np.float32)
    mask_image = np.zeros((mask_p.shape[0], image.shape[0], image.shape[1]), dtype=np.float32)
    mask_image = generate_mask(mask_p=mask_p, mask_image=mask_image)
    for i in range(mask_image.shape[0]):
        mask_all[0]= np.logical_or(mask_all[0], mask_image[i])


        mask_pos = [mask_image[1], mask_all[0]]
        mask_neg = [mask_image[3], mask_all[0]]


    mask_pos = np.asarray(mask_pos)
    mask_neg = np.asarray(mask_neg)
    return mask_pos, mask_neg

def generate_cover_tracking(mask_p):
    min_x = min(mask_p[:,0])
    w_x = max(mask_p[:,0]+mask_p[:,2])-min_x
    min_y = min(mask_p[:, 1])
    h_y = max(mask_p[:, 1] + mask_p[:, 3])-min_y
    cover_p = np.asarray([min_x,min_y, w_x,h_y])

    return cover_p

def generate_cover_PE(image,mask_p):
    cover_mask = np.zeros((1, image.shape[0], image.shape[1]), dtype=np.float32)
    min_x = min(mask_p[:,0])
    w_x = max(mask_p[:,0]+mask_p[:,2])-min_x
    min_y = min(mask_p[:, 1])
    h_y = max(mask_p[:, 1] + mask_p[:, 3])-min_y
    cover_p = np.asarray([[min_x,min_y, w_x,h_y]])
    cover_mask = generate_mask(mask_p = cover_p,mask_image=cover_mask)

    return cover_mask[0], [(min_x),(min_y),(w_x),(h_y)]


def mask_generate_1by1(image, mask_p):
    mask_all = np.zeros((1, image.shape[0], image.shape[1]), dtype=np.float32)
    mask_image = np.zeros((mask_p.shape[0], image.shape[0], image.shape[1]), dtype=np.float32)
    mask_image = generate_mask(mask_p=mask_p, mask_image=mask_image)
    for i in range(mask_image.shape[0]):
        mask_all[0]= np.logical_or(mask_all[0], mask_image[i])
        mask_pos = [mask_image[1], mask_all[0]]
        mask_neg = [mask_image[3], mask_all[0]]
    mask_pos = np.asarray(mask_pos)
    mask_neg = np.asarray(mask_neg)

    #mask_pos[mask_pos < 1] = 10e-2
    #mask_neg[mask_neg < 1] = 10e-2

    #mask_pos = mask_pos[:]*(1-10e-2)+10e-2
    #mask_neg = mask_neg[:]*(1-10e-2)+10e-2

    #np 6channel is not that clever it must to be torch process

    #img_t = np.transpose(image, (2,0,1))
    #img_pos = np.concatenate((img_t*mask_pos[0],img_t*mask_pos[1]))
    #img_neg = np.concatenate((img_t*mask_neg[0],img_t*mask_neg[1]))
    '''
    img_pos = np.zeros((6, image.shape[0], image.shape[1]), dtype=np.float32)
    img_neg = np.zeros((6, image.shape[0], image.shape[1]), dtype=np.float32)
    '''
    return mask_pos, mask_neg

def mask_product(K1,K2,image,cover_mask):
    mask_pos = torch.split(K1,1,dim=1)
    mask_neg = torch.split(K2,1,dim=1)
    pos =torch.mul(image,mask_pos[0])
    neg = torch.mul(image,mask_neg[0])
    #all = torch.mul(image,mask_pos[1])
    #K11 = torch.cat((pos,all),dim=1)
    #K22 = torch.cat((neg,all),dim=1)
    cover_image = torch.mul(image, cover_mask)
    return pos,neg, cover_image




def mask_add(image,mask_p=np.array([])):

    if(~mask_p.any()):
        #Cat mask in tensor dim[1]/ 0: batch, 1: channel, 2,3 : imgsize
        mask_all = torch.ones((image.shape[0],1,image.shape[2],image.shape[3]))
        mask_part = torch.ones((image.shape[0],1,image.shape[2],image.shape[3]))

        img_mask_added = torch.cat((image,mask_all),1)
        img_mask_added = torch.cat((img_mask_added, mask_part), 1)
        return img_mask_added
    else:
        #[4,4,20,20]
        mask_all = np.zeros((image.shape[0],1,image.shape[2],image.shape[3]),dtype=np.float32)

        mask_image = np.zeros((image.shape[0],mask_p.shape[1],image.shape[2],image.shape[3]),dtype=np.float32)
        mask_image = generate_mask(mask_p=mask_p, mask_image=mask_image)
        for i in range(mask_image.shape[0]):
            for j in range(mask_image.shape[1]):
                mask_all[i][0] = np.logical_or(mask_all[i][0], mask_image[i][j])
        mask_tensor = torch.cat((torch.from_numpy(mask_all),torch.from_numpy(mask_image)),1)
        mask_tensor = mask_tensor.to(torch.device("cuda:0"))
        mask_torch = torch.cat((image,mask_tensor),1)
        return mask_torch

def calcul_iob_mask(input, gt):
    target = np.tile(gt, (input.shape[0], 1))
    gab = (input[:, 2:] + target[:, 2:]) / 2 - abs(
        input[:, :2] + input[:, 2:] / 2 - target[:, :2] - target[:, 2:] / 2)
    zero_gab = np.zeros_like(gab)
    gab = np.maximum(gab, zero_gab)
    gab = np.minimum(gab, input[:, [2, 3]])
    gab = np.minimum(gab, target[:, [2, 3]])
    iob = (gab[:, 0] * gab[:, 1]) / (input[:, 2] * input[:, 3])
    return iob

def make_sample_cover2(gt, mask, pos_num=100, neg_num=100):
    out_range = 1.2  # out_range >= 1.0
    box = np.random.rand(2 * (pos_num + neg_num), 4)
    xy_gt = np.tile((mask[:2] + mask[2:] / 2), (box.shape[0], 1))
    wh_gt = np.tile(gt[2:], (box.shape[0], 1))
    wh_mask = np.tile(mask[2:], (box.shape[0], 1))
    box[:, 2:] = wh_gt
    box[:, :2] = xy_gt + box[:, :2] * (wh_mask * out_range + box[:, 2:]) - (
            wh_mask * (out_range - 1) * 0.5 + box[:, 2:])
    iob = calcul_iob_mask(box, mask)
    sample_neg = box[iob < 0.4]
    neg = sample_neg[:neg_num]
    np.random.shuffle(neg)
    neg = neg[0]
    return neg
def make_sample_cover(mask,w,h,n =10000):
    gt_w = mask[2]
    gt_h = mask[3]
    w_mini = gt_w / 4
    h_mini = gt_h / 4
    xy_box = np.random.uniform((0,0),((w-w_mini),(h-h_mini)),(n,2))
    wh_box = np.tile(mask[2:]/4, (n, 1))
    box = np.concatenate((xy_box,wh_box), axis=1)
    iob = calcul_iob_mask(box, mask)
    pos_box = box[iob == 1]
    neg_box = box[iob != 1]

    pos = pos_box[:3]
    neg = neg_box[:3]
    return pos,neg