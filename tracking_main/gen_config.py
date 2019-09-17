import os
import json
import numpy as np

def gen_config(seqs):

        seq_home = '../dataset/Vot2015'
        save_home = '../result_fig'
        result_home = '../result'
        
        seq_name = seqs.name
        img_dir = os.path.join(seq_home, seq_name, 'img')
        gt_path = os.path.join(seq_home, seq_name, 'groundtruth.txt')

        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list = [os.path.join(img_dir,x) for x in img_list]

        gt = np.loadtxt(gt_path,delimiter=',')
        init_bbox = gt[0]
        
        savefig_dir = os.path.join(save_home,seq_name)
        result_dir = os.path.join(result_home,seq_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)




