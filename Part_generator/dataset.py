import os
import os.path
import glob

import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import torch.utils.data as data
from util_coco import *

import cv2


def map_grid(xywh, gt, resize_size=448.0):
    xywh_grid = torch.zeros_like(xywh)
    factor = (resize_size / gt[2:]).expand_as(xywh[:, :2]).float()
    xywh_grid[:, :2] = (xywh[:, :2] - gt[:2].float()) * factor
    xywh_grid[:, 2:] = xywh[:, 2:] * factor

    return xywh_grid


def calcul_IOB_mask(input, gt):
    target = gt.expand_as(input)
    gab = (input[:, 2:] + target[:, 2:]) / 2 - abs(input[:, :2] + input[:, 2:]/2 - target[:, :2] - target[:, 2:]/2)
    gab = torch.max(gab, torch.zeros_like(gab))
    gab = torch.min(gab, input[:, [2, 3]])
    gab = torch.min(gab, target[:, [2, 3]]).float()
    result = (gab[:, 0]*gab[:, 1])/(input[:, 2]*input[:, 3]).float()
    return result > 0.8


def padding_img(x, y, w, h, img_size, random=True, scale=2.0):
    if random:
        r = torch.rand(4)
        crop_w = (w * (1 + scale * r[0])).long()
        crop_h = (h * (1 + scale * r[1])).long()
        crop_x = x + (w - crop_w)*r[2]
        crop_y = y + (h - crop_h)*r[3]
    else:
        crop_w = w * (1 + scale)
        crop_h = h * (1 + scale)
        crop_x = x + (w - crop_w)/2
        crop_y = y + (h - crop_h)/2

    crop_x, crop_y, crop_w, crop_h = crop_x.long(), crop_y.long(), crop_w.long(), crop_h.long()

    crop_w = torch.min(crop_w, img_size[1])
    crop_h = torch.min(crop_h, img_size[0])
    crop_x = torch.min(torch.max(crop_x, torch.tensor([0]))[0] + crop_w, img_size[1]) - crop_w
    crop_y = torch.min(torch.max(crop_y, torch.tensor([0]))[0] + crop_h, img_size[0]) - crop_h

    return crop_x, crop_y, crop_w, crop_h


class COCODataset(data.Dataset):
    image_size = 448
    label = []

    def __init__(self, root, data_dir, list_file, train, transform):
        print('data init')
        anns = '{}/annotations/instances_{}.json'.format(root, data_dir)
        self.root = root + data_dir
        self.train = train
        self.transform = transform
        self.fnames = []
        self.boxes = []
        self.mean = (123, 117, 104)  # RGB
        self.coco = COCO(anns)
        self.category = []
        self.anns=[]
        self.part_boxes = []

        if isinstance(list_file, list):
            list_file = './' + list_file[0]
        with open(list_file) as f:
            lines = f.readlines()
        for line in lines:
            splited = line.strip().split()
            num_boxes = 1
            box = []
            cat=[]
            anns=[]
            part_box=[]

            for i in range(num_boxes):
                cat.append(int(splited[1 + 5 * i]))
                anns.append(int(splited[2 + 5 * i]))
                x = float(splited[3 + 5 * i])
                y = float(splited[4 + 5 * i])
                x2 = float(splited[5 + 5 * i])
                y2 = float(splited[6 + 5 * i])
                # c = splited[5+5*i]
                box_broken = [x, y, x2, y2]
                box.append(box_broken)
                # label.append(int(c)+1)
            self.category.append(cat)
            self.anns.append(anns)
            self.fnames.append(splited[0])
            self.boxes.append(box)
            self.part_boxes.append([float(item) for item in splited[7:len(splited)]])
        self.num_samples = len(self.boxes)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        sample = self.boxes[idx]
        img = cv2.imread(os.path.join(self.root + '/' + fname))
        if not self.train:
            img_ori = self.transform(img)
        boxes = self.part_boxes[idx]
        sample = sample[0]
        sample = np.asarray(sample)

        if self.train:
            #img, boxes = self.random_flip(img, boxes)
            #img,boxes = self.randomScale(img,boxes)
            img = self.randomBlur(img)
            img_hsv = self.BGR2HSV(img)
            img_hsv = self.RandomBrightness(img_hsv)
            img_hsv = self.RandomHue(img_hsv)  # color
            img_hsv = self.RandomSaturation(img_hsv)
            #img,boxes = self.randomShift(img_hsv,boxes)
            #img,boxes = self.randomCrop(img_hsv,boxes)
            img = self.HSV2BGR(img_hsv)
            #img = self.randomBlur(img)
            #img = self.RandomBrightness(img)
            #img = self.RandomHue(img)
            #img = self.RandomSaturation(img)

        #boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)

        if not boxes:
            imgIds = int(fname[:-4])
            catIds=self.category[idx]
            annIds=self.anns[idx]
            #annIds = self.coco.getAnnIds(imgIds=imgIds,catIds=catIds)
            anns = self.coco.loadAnns(annIds)

            #srt = [(i, j['area']) for (i, j) in enumerate(anns) if
            #       (j['iscrowd'] == False) & (j['category_id'] == catIds[0]) & (j['area'] > 800) & (
            #       (j['area'] / (j['bbox'][2] * j['bbox'][3] + 10e-6) > 0.1)) & (j['bbox'][2] > 30) & (
            #                   j['bbox'][3] > 30)]
            #srt = sorted(srt, key=lambda a: a[1], reverse=True)
            mask_img = self.coco.annToMask(anns[0])
            partial_box = part_generate(mask_img,sample)
        else:
            partial_box = boxes
            partial_box = np.asarray(partial_box)
            partial_box = np.reshape(partial_box,(40,4))

        img = self.BGR2RGB(img)  # because pytorch pretrained model use RGB
        img = self.subMean(img, self.mean)

        #size = torch.tensor(img.shape[:2])
        #crop_x, crop_y, crop_w, crop_h = sample[0], sample[1], sample[2], sample[3]
        #if self.train:
            #crop_x, crop_y, crop_w, crop_h = padding_img(crop_x, crop_y, crop_w, crop_h, size,random=False)
        #else:
            #crop_x, crop_y, crop_w, crop_h = padding_img(crop_x, crop_y, crop_w, crop_h, size, random=False, scale=1.0)

        #img = img[crop_y:crop_y+(crop_h/2), crop_x:crop_x+crop_w]

        new_crop = generate_cover(img, sample, padding=True)
        new_crop = torch.FloatTensor(new_crop)
        partial_box =  torch.FloatTensor(partial_box)



        #new_img = img[new_crop[1]:(new_crop[1]+new_crop[3]), new_crop[0]:(new_crop[0]+new_crop[2]),:]


        '''dpi = 80.0
        size = img.shape[:2]
        figsize = (size[1] / dpi, size[0] / dpi)
        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img)'''

        #img_xywh = torch.tensor([crop_x,crop_y, crop_w,crop_h]).float()

            #if self.train == True :
        img, encoder_box = crop_maping(img, new_crop, partial_box)
        h, w, _ = img.shape
        encoder_box /= torch.Tensor([w,h,w,h]).expand_as(encoder_box)  # 7x7x10


        img = cv2.resize(img, (self.image_size, self.image_size))
        target = self.encoder(encoder_box)
        img = self.transform(img)
        if self.train:
            return img, target
        else:
            return img, img_ori,new_crop

    def encoder(self, boxes):
        # boxes (tensor) [[x,y,w,h],[]]
        # labels (tensor) [...]
        # return 7x7x30
        
        grid_num = 14
        target = torch.zeros((grid_num, grid_num, 10))
        cell_size = 1. / grid_num
        wh = boxes[:, 2:]
        cxcy = (boxes[:, 2:]/2 + boxes[:, :2])
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample / cell_size).ceil() - 1  #
            target[int(ij[1]), int(ij[0]), 4] = 1
            target[int(ij[1]), int(ij[0]), 9] = 1
            xy = ij * cell_size
            delta_xy = (cxcy_sample - xy) / cell_size
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
        return target

    def __len__(self):
        return self.num_samples

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def RandomBrightness(self, hsv):
        if random.random() < 0.5:
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.7, 1.3])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
        return hsv

    def RandomSaturation(self, hsv):
        if random.random() < 0.5:
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
        return hsv

    def RandomHue(self, hsv):
        if random.random() < 0.5:
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
        return hsv

    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def randomShift(self,bgr,boxes):
        center = (boxes[:,2:]+boxes[:,:2])/2
        if random.random() <0.5:
            height,width,c = bgr.shape
            after_shfit_image = np.zeros((height,width,c),dtype=bgr.dtype)
            after_shfit_image[:,:,:] = (104,117,123) #bgr
            shift_x = random.uniform(-width*0.2,width*0.2)
            shift_y = random.uniform(-height*0.2,height*0.2)
            #print(bgr.shape,shift_x,shift_y)
            if shift_x>=0 and shift_y>=0:
                after_shfit_image[int(shift_y):,int(shift_x):,:] = bgr[:height-int(shift_y),:width-int(shift_x),:]
            elif shift_x>=0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width-int(shift_x),:]
            elif shift_x <0 and shift_y >=0:
                after_shfit_image[int(shift_y):,:width+int(shift_x),:] = bgr[:height-int(shift_y),-int(shift_x):,:]
            elif shift_x<0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = bgr[-int(shift_y):,-int(shift_x):,:]

            shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:,0] >0) & (center[:,0] < width)
            mask2 = (center[:,1] >0) & (center[:,1] < height)
            mask = (mask1 & mask2).view(-1,1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if len(boxes_in) == 0:
                return bgr,boxes
            box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in+box_shift
            return after_shfit_image,boxes_in
        return bgr,boxes

    def randomScale(self,bgr,boxes):
        if random.random() < 0.5:
            scale = random.uniform(0.8,1.2)
            height,width,c = bgr.shape
            bgr = cv2.resize(bgr,(int(width*scale),height))
            scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr,boxes
        return bgr,boxes

    def randomCrop(self,bgr,boxes):
        if random.random() < 0.5:
            center = (boxes[:,2:]+boxes[:,:2])/2
            height,width,c = bgr.shape
            h = random.uniform(0.6*height,height)
            w = random.uniform(0.6*width,width)
            x = random.uniform(0,width-w)
            y = random.uniform(0,height-h)
            x,y,h,w = int(x),int(y),int(h),int(w)

            center = center - torch.FloatTensor([[x,y]]).expand_as(center)
            mask1 = (center[:,0]>0) & (center[:,0]<w)
            mask2 = (center[:,1]>0) & (center[:,1]<h)
            mask = (mask1 & mask2).view(-1,1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if(len(boxes_in)==0):
                return bgr,boxes
            box_shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:,0]=boxes_in[:,0].clamp_(min=0,max=w)
            boxes_in[:,2]=boxes_in[:,2].clamp_(min=0,max=w)
            boxes_in[:,1]=boxes_in[:,1].clamp_(min=0,max=h)
            boxes_in[:,3]=boxes_in[:,3].clamp_(min=0,max=h)

            img_croped = bgr[y:y+h,x:x+w,:]
            return img_croped,boxes_in
        return bgr,boxes


    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta,delta)
            im = im.clip(min=0,max=255).astype(np.uint8)
        return im

def generate_cover(image,all_box,padding=False):
    min_x = all_box[0].min()
    w_x = all_box[0].max()+all_box[2].max() - min_x
    min_y = all_box[1].min()
    h_y = all_box[1].max()+all_box[3].max() - min_y

    w=0.25*all_box[3]
    h=0.25*all_box[2]
    if (padding):
        min_x -= w
        w_x += 2 * w
        min_y -= h
        h_y += 2 * h
        if (min_x < 0):
            w_x+=min_x
            min_x = 0
        if (min_y < 0):
            h_y+=min_y
            min_y = 0
        if (min_x + w_x>image.shape[1]): w_x -= (min_x + w_x - image.shape[1])
        if(min_y + h_y>image.shape[0]): h_y -= (min_y + h_y - image.shape[0])


    cover_p = np.asarray([int(min_x),int(min_y), int(w_x),int(h_y)])

    return cover_p

def crop_maping(img,new_crop,boxes):
        center = (boxes[:, :2] + (boxes[:, 2:]/2))

        boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]

        #height, width, c = img.shape

        x, y, w, h = int(new_crop[0]), int(new_crop[1]), int(new_crop[2]), int(new_crop[3])


        center = center - torch.FloatTensor([[x, y]]).expand_as(center)
        mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
        mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
        mask = (mask1 & mask2).view(-1, 1)

        boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
        if (len(boxes_in) == 0):
            return img, boxes
        box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

        boxes_in = boxes_in - box_shift
        boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
        boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
        boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
        boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

        boxes_in[:, 2:] = boxes_in[:, 2:] -  boxes_in[:, :2]

        img_croped = img[y:y + h, x:x + w, :]

        return img_croped, boxes_in
