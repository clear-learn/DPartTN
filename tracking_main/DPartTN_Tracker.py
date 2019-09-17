import numpy as np
import os
import pickle
import sys
import cv2

import time
import argparse
import json

import PIL.Image as Image
from resnet_yolo import resnet50 as PG_resnet
import network_256concat_91 as network
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from model import *
from options import *
from sample_generator import *
from sample_genrator_new import *
from data_prov import *
from predict import *
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)


class DPartTN_tracker(object):
    def __init__(self, image, region):
        self.device0 = torch.device("cuda:0")
        self.target_bbox = region
        self.image_first = image
        self.PE_resnet = network.resnet().to(self.device0)
        load_state(self.PE_resnet, torch.load('/home/adrenaline36/Eval_2017/DPartTN_35/DPartTN/original.pth'))
        for param in self.PE_resnet.parameters():
            param.requires_grad = False
        self.PE_resnet.eval()
        self.PEnet =network.PE().to(self.device0)
        load_state(self.PEnet, torch.load('/home/adrenaline36/Eval_2017/DPartTN_35/DPartTN/best_256_concat.pth', map_location="cuda:0")['state_dict'])
        for param in self.PEnet.parameters():
            param.requires_grad = False
        self.PEnet.eval()
        self.PGnet = PG_resnet().to(self.device0)
        self.PGnet.load_state_dict(torch.load('/home/adrenaline36/Eval_2017/DPartTN_35/DPartTN/COCO_best_sia.pth'))
        self.PGnet.eval().cuda(self.device0)

        self.PG_candidate, new_gt = self.PG_mdnet(self.PGnet, self.image_first, self.target_bbox)
        self.PG_candidate = self.PG_candidate.numpy()
        self.model =MDNet('/home/adrenaline36/Eval_2017/DPartTN_35/DPartTN/models/mdnet_imagenet_vid.pth',self.PG_candidate.shape[0]+1)
        self.model.to(self.device0)
        self.model.set_learnable_params(opts['ft_layers'])
        self.criterion = BinaryLoss()
        self.init_optimizer = self.set_optimizer(self.model, opts['lr_init'],{'fc6': 10})
        self.update_optimizer = self.set_optimizer(self.model, opts['lr_update'],{'fc6': 10})

        #pos_examples = gen_samples(SampleGenerator('gaussian', self.image_first.size, 0.1, 1.3),
        #                           self.target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])
        pos_examples = SampleGenerator('gaussian', self.image_first.size, 0.1, 1.3)(
            self.target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

        if pos_examples.shape[0] == 0:
            self.target_bbox[2:] = 10
            pos_examples =SampleGenerator('gaussian', self.image_first.size, 0.1, 1.2)(
                                       self.target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])
        neg_examples = np.concatenate([
            SampleGenerator('uniform', image.size, 1, 1.6)(
                self.target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
            SampleGenerator('whole', image.size)(
                self.target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
        neg_examples = np.random.permutation(neg_examples)
        #neg_examples = np.concatenate([
        #    gen_samples(SampleGenerator('uniform', self.image_first.size, 1, 2, 1.1),
        #                self.target_bbox, opts['n_neg_init'] // 2, opts['overlap_neg_init']),
        #    gen_samples(SampleGenerator('whole', self.image_first.size, 0, 1.2, 1.1),
        #                self.target_bbox, opts['n_neg_init'] // 2, opts['overlap_neg_init'])])
        #neg_examples = np.random.permutation(neg_examples)
        # Extract pos/neg features
        pos_feats = self.forward_samples(self.model, self.image_first, pos_examples)
        neg_feats = self.forward_samples(self.model, self.image_first, neg_examples)
        self.train(self.model, self.criterion, self.init_optimizer, pos_feats, neg_feats, opts['maxiter_init'],
                   in_layer='fc4', K=0)
        if self.PG_candidate.shape[0] > 1:
            for branch in range(self.PG_candidate.shape[0]):
                pos_examples = gen_samples_new(self.image_first, self.PG_candidate[branch, :], 10, self.target_bbox,
                                               numk=True, image_size=self.image_first.size)
                neg_examples = gen_samples_new(self.image_first, self.PG_candidate[branch, :], 30, self.target_bbox,
                                               numk=False, image_size=self.image_first.size)

                # Extract pos/neg features

                pos_feats_branch = self.forward_samples(self.model, self.image_first, pos_examples)
                neg_feats_branch = self.forward_samples(self.model, self.image_first, neg_examples)
                # feat_dim_branch = pos_feats_branch.size(-1)

                self.train(self.model, self.criterion, self.init_optimizer, pos_feats_branch, neg_feats_branch, 3,
                           in_layer='fc4', K=branch + 1)

        self.target_sample_generator = SampleGenerator('gaussian', self.image_first.size, opts['trans_f'],
                                                       opts['scale_f'],
                                                       valid=True)

        # sample_generator = SampleGenerator('sample', image.size, opts['trans_f'], opts['scale_f'], valid=True)
        self.pos_generator = SampleGenerator('gaussian', self.image_first.size, 0.1, 1.2)
        self.neg_generator = SampleGenerator('uniform', self.image_first.size, 1.5, 1.2)

        self.pos_feats_all = [pos_feats[:opts['n_pos_update']]]
        self.neg_feats_all = [neg_feats[:opts['n_neg_update']]]
        self.feat_dim = pos_feats.size(-1)

        self.cover_box = np.copy(self.target_bbox)

    def track(self, image):
        # Image starts with 0th image

        self.target_bbox_list = []
        self.score_list = []

        if self.PG_candidate.shape[0] > 1:
            for fc_index in range(self.PG_candidate.shape[0]):
                #target_samples = gen_samples(self.target_sample_generator, self.PG_candidate[fc_index, :], 30)
                target_samples = self.target_sample_generator(self.PG_candidate[fc_index, :], 30)

                target_sample_scores = self.forward_samples(self.model, image, target_samples, out_layer='fc6',
                                                            branch=fc_index + 1)
                target_top_scores, target_top_idx = target_sample_scores[:, 1].topk(5)
                target_top_idx = target_top_idx.cpu().numpy()
                target_top_scores = target_top_scores.cpu().numpy()
                target_bbox_part = target_samples[target_top_idx].mean(axis=0)
                target_top_scores = target_top_scores.mean(axis=0)

                self.target_bbox_list.append(target_bbox_part)
                self.score_list.append(target_top_scores)

        if self.PG_candidate.shape[0] > 1:
            self.PG_candidate = np.asarray(self.target_bbox_list)
            self.top_scores = np.asarray(self.score_list)
            _, where_nms = self.NMS_BB_mdnet(torch.FloatTensor(self.PG_candidate), torch.FloatTensor(self.top_scores))
            where_nms = np.where(where_nms.cpu().numpy())
            where_nms = where_nms[0]
            self.delete_model(self.model, (where_nms + 1))
            self.PG_candidate = np.delete(self.PG_candidate, where_nms, 0)
            _, where = self.PE_mdnet(self.PEnet, self.PE_resnet, image, self.PG_candidate,
                                     np.array([0, 0, image.size[0], image.size[1]]))
            where = where[0]
            self.PG_candidate = np.delete(self.PG_candidate, where, 0)
            self.delete_model(self.model, (where + 1))

        #success_TorF_samples = gen_samples(self.target_sample_generator, self.target_bbox, opts['n_samples'])
        success_TorF_samples = self.target_sample_generator(self.target_bbox, opts['n_samples'])

        success_TorF_scores = self.forward_samples(self.model, image, success_TorF_samples, out_layer='fc6', branch=0)
        success_TorF_scores, success_TorF_idx = success_TorF_scores[:, 1].topk(5)
        success_TorF_idx = success_TorF_idx.cpu().numpy()
        success_TorF_scores = success_TorF_scores.cpu().numpy()
        self.target_bbox = success_TorF_samples[success_TorF_idx].mean(axis=0)
        success_TorF = success_TorF_scores.mean(axis=0)

        success_TorF_swich = success_TorF > 0

        if self.PG_candidate.shape[0] > 1:
            self.cover_box = generate_cover_tracking(self.PG_candidate, image)
        else:
            self.cover_box = np.copy(self.target_bbox)

        cover_samples = np.expand_dims(self.cover_box, axis=0)
        cover_samples_scores = self.forward_one_samples(self.model, image, cover_samples, out_layer='fc6', branch=0)
        cover_box_top_scores = cover_samples_scores[:, 1]
        cover_box_top_scores = cover_box_top_scores.cpu().numpy()
        switch = cover_box_top_scores > 0.5

        if switch == False:
            self.target_sample_generator.expand_trans(1.5)

            self.cover_box = np.copy(self.target_bbox)
            where_branch = np.arange(1, len(self.model.branches), 1)
            self.delete_model(self.model, where_branch)
            self.PG_candidate, new_cover = self.PG_mdnet(self.PGnet, image, self.cover_box)
            self.PG_candidate = self.PG_candidate.numpy()
            if self.PG_candidate.shape[0] > 0:
                add_K = self.PG_candidate.shape[0]
                self.add_model(self.model, add_K)
                self.train_branch(self.model, 1, self.PG_candidate, image, self.criterion, self.cover_box)
        else:
            self.target_sample_generator.set_trans(0.6)

        if success_TorF_swich == False:
            pos_examples = self.pos_generator(self.target_bbox,
                                       opts['n_pos_update'],
                                       opts['overlap_pos_update'])
            if pos_examples.shape[0]==0:
                self.target_bbox[2:] = 10
                pos_examples = self.pos_generator(self.target_bbox,
                                           opts['n_pos_update'],
                                           opts['overlap_pos_update'])

            neg_examples = self.neg_generator(self.target_bbox,
                                       opts['n_neg_update'],
                                       opts['overlap_neg_update'])

            # Extract pos/neg features
            pos_feats = self.forward_samples(self.model, image, pos_examples)
            neg_feats = self.forward_samples(self.model, image, neg_examples)
            self.pos_feats_all.append(pos_feats)
            self.neg_feats_all.append(neg_feats)
            if len(self.pos_feats_all) > opts['n_frames_long']:
                del self.pos_feats_all[0]
            if len(self.neg_feats_all) > opts['n_frames_short']:
                del self.neg_feats_all[0]

            # Short term update
            nframes = min(opts['n_frames_short'], len(self.pos_feats_all))
            pos_data = torch.stack(self.pos_feats_all[-nframes:], 0).view(-1, self.feat_dim)
            neg_data = torch.stack(self.neg_feats_all, 0).view(-1, self.feat_dim)
            self.train(self.model, self.criterion, self.update_optimizer, pos_data, neg_data, opts['maxiter_update'],
                       in_layer='fc4',
                       K=0)

        return self.cover_box

    def NMS_BB_mdnet(self, bboxes, scores=torch.zeros(0), threshold=0.5):
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 0] + bboxes[:, 2]
        y2 = bboxes[:, 1] + bboxes[:, 3]

        if scores.shape[0] != bboxes.shape[0]:
            scores = torch.Tensor(range(bboxes.shape[0]))

        _, order = scores.sort(0, descending=True)
        # keep.append(order[range(len(order))])
        # delete = []
        check = torch.ones_like(order)

        for idx in range(len(order) - 1):
            i = order[idx]
            if (check[i] == True):
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
                    j = ids[jdx].item()
                    if (i != j):
                        check[j] = 0
        return check, 1 - check

    def delete_model(self, model, k):
        K_bran = np.sort(k)[::-1]
        model.delete_branch(K_bran)

    def img_preper(self, images, sample):
        transform = transforms.ToTensor()
        sample = sample.cpu().numpy()
        images = np.array(images)
        mean = (123, 117, 104)
        img = self.subMean(images, mean)
        new_crop = self.generate_cover(images, sample)
        # new_crop = np.asarray([int(sample[0])-5,int(sample[1])-5,int(sample[2])+5,int(sample[3]+5)])

        crop_x2 = new_crop[0] + new_crop[2]
        crop_y2 = new_crop[1] + new_crop[3]
        img = img[new_crop[1]:crop_y2, new_crop[0]:crop_x2, :]
        img = cv2.resize(img, (448, 448))
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        new_GT = new_crop.astype(np.float32)
        new_GT = torch.from_numpy(new_GT)

        return img, new_GT

    def Preper_dataset(self, img, part, gt):
        transform = transforms.ToTensor()
        img = np.asarray(img)
        img = img.astype(np.uint16)
        img = img.astype(np.float32)
        img = img / 255.
        img_mask_list = []

        all_box = part
        cov = gt
        img_cov = img[int(cov[1]):int(cov[1] + cov[3]),
                  int(cov[0]):int(cov[0] + cov[2])][:]

        for i in range(part.shape[0]):
            mask = np.ones((img.shape[0], img.shape[1]), dtype=np.int16)
            mask[int(part[i, 1]):int(part[i, 1] + part[i, 3]),
            int(part[i, 0]):int(part[i, 0] + part[i, 2])] = 0
            img_mask = [img[:, :, 0] * mask, img[:, :, 1] * mask, img[:, :, 2] * mask]

            # for dim_idx in range(3):
            # img_mask[dim_idx][img_mask[dim_idx] == 0.] = 0.

            img_mask = np.transpose(img_mask, (1, 2, 0))[int(cov[1]):int(cov[1] + cov[3]),
                       int(cov[0]):int(cov[0] + cov[2])][:]
            img_mask_resize = cv2.resize(img_mask, (224, 224), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
            img_mask_tran = transform(img_mask_resize)
            img_mask_list.append(img_mask_tran)
        img_mask_tensor = torch.stack(img_mask_list)

        whole_mask = np.ones((img.shape[0], img.shape[1]), dtype=np.int16)
        for it in range(all_box.shape[0]):
            whole_mask[int(all_box[it][1]):int(all_box[it][1] + all_box[it][3]),
            int(all_box[it][0]):int(all_box[it][0] + all_box[it][2])] = 0

        # img_whole = [img[:, :, 0] * whole_mask, img[:, :, 1] * whole_mask, img[:, :, 2] * whole_mask]

        # for dim_idx in range(3):
        # img_whole[dim_idx][img_whole[dim_idx] == 0.] = 0.

        # img_whole_transpose = np.transpose(img_whole, (1, 2, 0))[int(cov[1]):int(cov[1] + cov[3]),
        # int(cov[0]):int(cov[0] + cov[2])][:]
        cover_mask = cv2.resize(img_cov, (224, 224), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
        # whole_mask = cv2.resize(img_whole_transpose, (224, 224), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)

        cover_mask = transform(cover_mask)
        # whole_mask = transform(whole_mask)

        cover_mask = cover_mask.unsqueeze(0).repeat(img_mask_tensor.shape[0], 1, 1, 1)
        # whole_mask = whole_mask.unsqueeze(0).repeat(img_mask_tensor.shape[0], 1, 1, 1)

        return cover_mask, img_mask_tensor, cov

    def PE_mdnet(self, PEnet, PE_resnet, images, part, gt):
        PE_resnet.eval()
        PEnet.eval()
        cover, part, cov = self.Preper_dataset(images, part, gt)
        cover = cover.cuda(self.device0)
        part = part.cuda(self.device0)
        # whole = whole.cuda()
        # feature3_w = PE_resnet(whole)
        feature3 = PE_resnet(cover)
        feature_pos3 = PE_resnet(part)
        output = PEnet(feature3, feature_pos3)
        score = torch.nn.functional.sigmoid(output)
        score = (score[:, 1] - score[:, 0]) > 0.8
        score = score.cpu().numpy()

        # score = output[:, 0] > output[:, 1]
        where = np.where(score)
        return score, where

    def PG_mdnet(self, PGnet, images, gt, add=False, number=0):
        PGnet.eval()
        gt = gt.astype(np.float32)
        gt = torch.from_numpy(gt)
        img, new_gt = self.img_preper(images, gt)
        img = img.cuda(self.device0)

        pred = PGnet(img)

        preds = decoder(pred, new_gt, thresh=add, number=number)

        if add == True:
            return preds, new_gt
        else:
            return preds, new_gt

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def generate_cover(self, image, all_box, padding=True):
        # cover_mask = np.zeros((1, image.shape[0], image.shape[1]), dtype=np.float32)

        min_x = all_box[0].min()
        w_x = all_box[0].max() + all_box[2].max() - min_x
        min_y = all_box[1].min()
        h_y = all_box[1].max() + all_box[3].max() - min_y

        w = 0.25 * all_box[3]
        h = 0.25 * all_box[2]
        if (padding):
            min_x -= w
            w_x += 2 * w
            min_y -= h
            h_y += 2 * h
            if (min_x < 0):
                w_x += min_x
                min_x = 0
            if (min_y < 0):
                h_y += min_y
                min_y = 0
            if (min_x + w_x > image.shape[1]): w_x -= (min_x + w_x - image.shape[1])
            if (min_y + h_y > image.shape[0]): h_y -= (min_y + h_y - image.shape[0])

        cover_p = np.asarray([int(min_x), int(min_y), int(w_x), int(h_y)])
        # cover_mask = generate_mask(mask_p = cover_p,mask_image=cover_mask)

        return cover_p

    def set_optimizer(self, model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'],
                      w_decay=opts['w_decay']):
        params = model.get_learnable_params()
        param_list = []
        for k, p in params.items():
            lr = lr_base
            for l, m in lr_mult.items():
                if k.startswith(l):
                    lr = lr_base * m
            param_list.append({'params': [p], 'lr': lr})
        optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
        return optimizer

    def forward_samples(self, model, image, samples, out_layer='conv3', branch=0):
        self.model.eval()
        extractor = RegionExtractor(image, samples, opts['img_size'], opts['padding'], opts['batch_test'])
        for i, regions in enumerate(extractor):
            regions = regions.cuda(self.device0)
            feat = model(regions, k=branch, out_layer=out_layer)
            if i == 0:
                feats = feat.data.clone()
            else:
                feats = torch.cat((feats, feat.data.clone()), 0)
        return feats

    def train(self, model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4', K=0):
        self.model.train()

        batch_pos = 500
        batch_neg = 1500
        # batch_test = opts['batch_test']
        batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

        pos_idx = np.random.permutation(pos_feats.size(0))
        neg_idx = np.random.permutation(neg_feats.size(0))
        while (len(pos_idx) < batch_pos * maxiter):
            pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
        while (len(neg_idx) < batch_neg_cand * maxiter):
            neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
        pos_pointer = 0
        neg_pointer = 0

        for iter in range(maxiter):
            # select pos idx
            pos_next = pos_pointer + batch_pos
            pos_cur_idx = pos_idx[pos_pointer:pos_next]
            pos_cur_idx = pos_feats.new(pos_cur_idx).long()
            pos_pointer = pos_next

            # select neg idx
            neg_next = neg_pointer + batch_neg_cand
            neg_cur_idx = neg_idx[neg_pointer:neg_next]
            neg_cur_idx = neg_feats.new(neg_cur_idx).long()
            neg_pointer = neg_next

            # create batch
            batch_pos_feats = pos_feats[pos_cur_idx]
            batch_neg_feats = neg_feats[neg_cur_idx]

            # hard negative mining
            '''if batch_neg_cand > batch_neg:
                model.eval()
                for start in range(0, batch_neg_cand, batch_test):
                    end = min(start + batch_test, batch_neg_cand)
                    score = model(batch_neg_feats[start:end],K, in_layer=in_layer)
                    if start == 0:
                        neg_cand_score = score.data[:, 1].clone()
                    else:
                        neg_cand_score = torch.cat((neg_cand_score, score.data[:, 1].clone()), 0)

                _, top_idx = neg_cand_score.topk(batch_neg)
                batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
                model.train()'''

            # forward

            pos_score = model(batch_pos_feats, k=K, in_layer=in_layer)
            neg_score = model(batch_neg_feats, k=K, in_layer=in_layer)

            # optimize
            loss = criterion(pos_score, neg_score)
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
            optimizer.step()

            # print "Iter %d, Loss %.4f" % (iter, loss.data[0])

    def add_model(self, model, k):
        model.plus_branch(k)

    def train_branch(self, model, k, boxes, image, criterion, c_box):
        model.to(self.device0)
        if opts['use_gpu']:
            model = model.cuda(self.device0)
        model.set_learnable_params(opts['ft_layers'])
        init_optimizer = self.set_optimizer(model, opts['lr_init'])

        for branch in range(k, boxes.shape[0]):
            pos_examples = gen_samples_new(image, boxes[branch, :], 10, c_box, numk=True, image_size=image.size)
            neg_examples = gen_samples_new(image, boxes[branch, :], 30, c_box, numk=False, image_size=image.size)

            pos_feats_branch = self.forward_samples(model, image, pos_examples)
            neg_feats_branch = self.forward_samples(model, image, neg_examples)

            self.train(model, criterion, init_optimizer, pos_feats_branch, neg_feats_branch, 3,
                       in_layer='fc4', K=branch + 1)

    def forward_one_samples(self, model, image, samples, out_layer='conv3', branch=0):
        model.eval()
        extractor = RegionExtractor(image, samples, opts['img_size'], opts['padding'], 1)
        for i, regions in enumerate(extractor):
            regions = regions.cuda(self.device0)
            feat = model(regions, k=branch, out_layer=out_layer)
            if i == 0:
                feats = feat.data.clone()
            else:
                feats = torch.cat((feats, feat.data.clone()), 0)
        return feats