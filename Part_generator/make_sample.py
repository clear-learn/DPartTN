import torch


def calcul_iob_mask(input, gt):
    target = gt.expand_as(input).float()
    gab = (input[:, 2:] + target[:, 2:]) / 2 - abs(input[:, :2] + input[:, 2:]/2 - target[:, :2] - target[:, 2:]/2)
    gab = torch.max(gab, torch.zeros_like(gab))
    gab = torch.min(gab, input[:, [2, 3]])
    gab = torch.min(gab, target[:, [2, 3]]).float()
    iob = (gab[:, 0]*gab[:, 1])/(input[:, 2]*input[:, 3]).float()
    return iob


def make_sample(gt, pos_num=100, neg_num=100):
    grid = 1/7.
    out_range = 1.5  # out_range >= 1.0
    box = torch.rand(2*(pos_num+neg_num), 4)
    xy_gt = (gt[:2] + gt[2:]*0.5).expand_as(box[:, :2]).float()
    wh_gt = gt[2:].expand_as(box[:, 2:]).float()
    box[:, 2:] = (box[:, 2:] + 0.5) * grid * wh_gt
    box[:, :2] = xy_gt + box[:, :2] * (wh_gt*out_range + box[:, 2:]) - (wh_gt*(out_range-1)*0.5 + box[:, 2:])
    iob = calcul_iob_mask(box, gt)
    sample_pos = box[0.6 < iob]
    sample_neg = box[iob < 0.4]

    while sample_pos.size()[0] < pos_num or sample_neg.size()[0] < neg_num:
        box = torch.rand(2*(pos_num+neg_num), 4)
        box[:, 2:] = (box[:, 2:] + 0.5) * grid * wh_gt
        box[:, :2] = xy_gt + box[:, :2] * (wh_gt + box[:, 2:]) - box[:, 2:]
        iob = calcul_iob_mask(box, gt)
        if sample_pos.size()[0] < pos_num:
            sample_pos = torch.cat((sample_pos, box[0.5 <= iob]), 0)
        if sample_neg.size()[0] < neg_num:
            sample_neg = torch.cat((sample_neg, box[iob < 0.5]), 0)

    return sample_pos[:pos_num], sample_neg[:neg_num]

def mask_generator(gt):
    pos,neg = make_sample(gt)