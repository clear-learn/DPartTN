import os
import os.path
#from util import *
import torch.utils.data as data
#from util.util_mask_1by1 import *
from pycocotools.coco import COCO
import cv2
from util_mask_1by1 import *
from util_train import *
from util import *



class COCODataset(data.Dataset):
    image_size = 224
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
        # sample = self.masks[idx]
        img = cv2.imread(os.path.join(self.root + '/' + fname))
        boxes = self.boxes[idx]
        parts = self.part_boxes[idx]
        # boxlist = []
        mask = boxes[0]
        mask = [mask[0], mask[1], mask[2], mask[3]]
        mask = np.asarray(mask)

        if not parts:

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
            partial_box = part_generate(mask_img,boxes[0])
            neg = neg_generate(mask_img, boxes[0])
            pos = partial_box[0:14]
            np.random.shuffle(pos)
            neg = neg[0:14]
        else:
            partial_box = parts
            partial_box = np.asarray(partial_box)
            partial_box = np.reshape(partial_box,(40,4))

            annIds = self.anns[idx]
            anns = self.coco.loadAnns(annIds)
            mask_img = self.coco.annToMask(anns[0])
            neg = neg_generate(mask_img, boxes[0])
            pos = partial_box[0:14]
            np.random.shuffle(pos)
            neg = neg[0:14]

        h, w, _ = img.shape

        img = self.BGR2RGB(img)

        img = img.astype(np.uint16)
        img = img.astype(np.float32)
        img = img / 255.

        boxen_pos = pos[0]
        boxen_neg = neg[0]

        all_box = np.concatenate((pos[4:], neg[1:]))
        np.random.shuffle(all_box)
        all_box = np.concatenate((pos[:4], neg[:1], all_box))
        all_box = all_box[0:14]

        min_box = [all_box[:, 0].min(), all_box[:, 1].min(), 1, 1]
        max_box = [all_box[:, 0].max() + all_box[:, 2].max(), all_box[:, 1].max() + all_box[:, 3].max(), 1, 1]
        cov = np.concatenate(([np.asarray(min_box)], [np.asarray(max_box)]))
        n, cov = generate_cover(img, cov)
        img_cov = img[int(cov[1]):int(cov[1] + cov[3]),
                  int(cov[0]):int(cov[0] + cov[2])][:]


        label = np.random.uniform(0, 1, 2)
        label = label > np.flip(label)
        label = torch.from_numpy(label.astype(np.float32))
        if (label[0] == True):
            pos_mask = np.ones((img.shape[0], img.shape[1]), dtype=np.int16)
            pos_mask[int(boxen_pos[1]):int(boxen_pos[1] + boxen_pos[3]),
            int(boxen_pos[0]):int(boxen_pos[0] + boxen_pos[2])] = 0
            img_pos = [img[:, :, 0] * pos_mask, img[:, :, 1] * pos_mask, img[:, :, 2] * pos_mask]
            img_pos = np.transpose(img_pos, (1, 2, 0))[int(cov[1]):int(cov[1] + cov[3]),
                      int(cov[0]):int(cov[0] + cov[2])][:]
            total_mask_pos = cv2.resize(img_pos, (224, 224), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
            cover_mask = cv2.resize(img_cov, (224, 224), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
            total_mask_pos = self.transform(total_mask_pos)
            cover_mask = self.transform(cover_mask)
            cover = cover_mask
            part = total_mask_pos
        else:
            neg_mask = np.ones((img.shape[0], img.shape[1]), dtype=np.int16)
            neg_mask[int(boxen_neg[1]): int(boxen_neg[1] + boxen_neg[3]),
            int(boxen_neg[0]): int(boxen_neg[0] + boxen_neg[2])] = 0
            img_neg = [img[:, :, 0] * neg_mask, img[:, :, 1] * neg_mask, img[:, :, 2] * neg_mask]
            img_neg = np.transpose(img_neg, (1, 2, 0))[int(cov[1]):int(cov[1] + cov[3]),
                      int(cov[0]):int(cov[0] + cov[2])][:]
            total_mask_neg = cv2.resize(img_neg, (224, 224), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
            cover_mask = cv2.resize(img_cov, (224, 224), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
            total_mask_neg = self.transform(total_mask_neg)
            cover_mask = self.transform(cover_mask)
            cover = cover_mask

            part = total_mask_neg

        img = cv2.resize(img, (224, 224), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
        img = self.transform(img)
        return img, cover, part, label

    def __len__(self):
        return self.num_samples
    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr