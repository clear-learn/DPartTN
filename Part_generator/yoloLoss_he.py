import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class yoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(yoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def calcul_iob(self, input, target):
        gab = (input[:, 2:] + target[:, 2:]) / 2 - abs(input[:, :2] + input[:, 2:]/2 - target[:, :2] - target[:, 2:]/2)
        gab = torch.max(gab, torch.zeros_like(gab))
        gab = torch.min(gab, input[:, [2, 3]])
        gab = torch.min(gab, target[:, [2, 3]])

        return gab[:, 0]*gab[:, 1]/(input[:, 2]*input[:, 3])

    def forward(self, pred_tensor, target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5=10) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,10)
        '''
        N = pred_tensor.size()[0]
        coo_mask = target_tensor[:, :, :, 4] > 0
        noo_mask = target_tensor[:, :, :, 4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        coo_pred = pred_tensor[coo_mask].view(-1, 10)
        box_pred = coo_pred.contiguous().view(-1, 5)  # box[x1,y1,w1,h1,c1]
        
        coo_target = target_tensor[coo_mask].view(-1, 10)
        box_target = coo_target.contiguous().view(-1, 5)

        # compute not contain obj loss
        noo_pred = pred_tensor[noo_mask].view(-1, 10)
        noo_target = target_tensor[noo_mask].view(-1, 10)
        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size())
        noo_pred_mask.zero_()
        noo_pred_mask[:, [4, 9]] = 1
        noo_pred_c = noo_pred[noo_pred_mask]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, size_average=False)

        # compute contain obj loss
        coo_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_response_mask.zero_()
        for i in range(0, box_target.size()[0], 2):  # choose the best iou box
            box1 = box_pred[i:i+2]
            box1_xywh = Variable(torch.FloatTensor(box1.size()))
            box1_xywh[:, :2] = box1[:, :2]/14. - 0.5*box1[:, 2:4]
            box1_xywh[:, 2:4] = box1[:, 2:4]
            box2 = box_target[i].view(-1, 5)
            box2_xywh = Variable(torch.FloatTensor(box2.size()))
            box2_xywh[:, :2] = box2[:, :2]/14. - 0.5*box2[:, 2:4]
            box2_xywh[:, 2:4] = box2[:, 2:4]
            iob = self.calcul_iob(box1_xywh[:, :4], box2_xywh[:, :4])  # [2,1]
            max_iou, max_index = iob.max(0)
            max_index = max_index.data.cuda()
            
            coo_response_mask[i+max_index] = 1

        # response loss
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)
        box_target_response = box_target[coo_response_mask].view(-1, 5)
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response[:, 4], size_average=False)
        xy_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], size_average=False)
        wh_loss = F.mse_loss(torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]), size_average=False)
        loc_loss = xy_loss + wh_loss

        return (self.l_coord*loc_loss + contain_loss + self.l_noobj*nooobj_loss)/N




