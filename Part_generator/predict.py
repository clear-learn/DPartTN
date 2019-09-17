import torch
import numpy as np


def map_ori_img(xywh, gt, crop_size=448):
    xywh_img = torch.zeros_like(xywh)
    xywh_img[:2] = gt[:2] + xywh[:2] * gt[2:] / crop_size
    xywh_img[2:] = xywh[2:] * gt[2:] / crop_size

    return xywh_img


def map_crop_img(xywh, crop_size=448):
    return xywh*crop_size


def decoder(predict, gt, thresh=False,number=0):
    pred= predict[0].cpu()
    gt = gt[0]
    grid_num = 14
    boxes = []
    probs = []
    cell_size = 1./grid_num


    contain = torch.argmax(pred[:, :, [4, 9]], dim=2)

    for i in range(grid_num):
        for j in range(grid_num):
            max_index = contain[i, j]
            if pred[i, j, 4+max_index*5] > 0:
                box = pred[i, j, max_index*5:max_index*5+4]
                x = (box[0] + j)*cell_size - box[2]/2
                y = (box[1] + i)*cell_size - box[3]/2
                w = box[2]
                h = box[3]

                xywh = torch.tensor([x, y, w, h])
                xywh = map_crop_img(xywh)
                xywh = map_ori_img(xywh, gt.float())
                boxes.append(xywh.view(1, -1))
                probs.append(pred[i, j, 4+max_index*5].view(1, -1))

    boxes = torch.cat(boxes, 0)  # (n,4)
    probs = torch.cat(probs).view(-1)  # (n,)



    if thresh:
        indxing = probs >= 0.4
        boxes = boxes[indxing]
        total=np.asarray(range(boxes.shape[0]))
        np.random.shuffle(total)
        boxes = boxes[total[:number],:]
        return boxes
    else:
        indxing = probs >= 0.7
        boxes=boxes[indxing]
        #check,_=NMS_BB_mdnet(boxes,probs[indxing])
        #boxes = boxes[check]
        #boxes = boxes[indxing]
        return boxes


def NMS_BB_mdnet(bboxes, scores=torch.zeros(0), threshold=0.4):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 0] + bboxes[:, 2]
    y2 = bboxes[:, 1] + bboxes[:, 3]

    if scores.shape[0]!=bboxes.shape[0]:
        scores = torch.Tensor(range(bboxes.shape[0]))

    _, order = scores.sort(0, descending=True)
    #keep.append(order[range(len(order))])
    #delete = []
    check = torch.ones_like(order)

    list_boxs = []

    for idx in range(len(order)-1):
        i=order[idx]
        if(check[i]==True):
            xx1 = x1[order].clamp(min=x1[i])
            xx2 = x2[order].clamp(max=x2[i])

            yy1 = y1[order].clamp(min=y1[i])
            yy2 = y2[order].clamp(max=y2[i])
            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h
            area = torch.mul(x2 - x1, y2 - y1)
            ovr = inter / (area[i] + area[order] - inter)
            ids = order[(ovr > threshold).nonzero()]


            for jdx in range(len(ids)):
                j=ids[jdx].item()
                if(i!=j):
                    check[j]=0

    return check, 1 - check


def nms(bboxes,scores,threshold=0.5):


    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,0] - bboxes[:,2]
    y2 = bboxes[:,1] - bboxes[:,3]

    areas = (x2 - x1) * (y2 - y1)

    _,order = scores.sort(0,descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)

def generate_cover_tracking(mask_p,img):
    min_x = min(mask_p[:,0])
    w_x = max(mask_p[:,0]+mask_p[:,2])-min_x
    min_y = min(mask_p[:, 1])
    h_y = max(mask_p[:, 1] + mask_p[:, 3])-min_y
    if (min_x < 0):
        w_x += min_x
        min_x = 0
    if (min_y < 0):
        h_y += min_y
        min_y = 0
    if (min_x + w_x > img.size[0]): w_x -= (min_x + w_x - img.size[0])
    if (min_y + h_y > img.size[1]): h_y -= (min_y + h_y - img.size[1])
    cover_p = np.asarray([min_x,min_y, w_x,h_y],dtype=np.float32)


    return cover_p

def load_state(net, param):
    model_dict = net.state_dict()
    for name, param in param.items():
        if name not in model_dict:
            continue
        #if name == 'conv1.weight':
        #    continue
        else:
            model_dict[name].copy_(param)