'''
Descripttion: 
version: 
Author: Jinlong Li CSU PhD
Date: 2024-07-10 21:02:00
LastEditors: Jinlong Li CSU PhD
LastEditTime: 2024-07-12 14:49:29
'''
import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
import torch


###---------------------------------->  data path
daytime_img_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset/train/daytime/samples/CAM_FRONT"
depth_resnet101_img_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_depth_resnet101/trainval-1.0/samples/CAM_FRONT_depth_resnet101"  ###png 

json_full_path = "/home/jinlongli/personal/DATASet/Nuscene_full/fakenightanddepth2day.json" #TODO:


# depth_lidar_img_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_lidar2depth/train/samples/CAM_FRONT"  ###jpg
# fake_night_iccv21_img_root = "/home/jinlongli/personal/DATASet/Nuscene_full/1.low_dark_fixed/daytime/samples/CAM_FRONT"  ###jpg
# fake_night_cyclegan_img_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_faked/train/2.cyclegan_dark/daytime/samples/CAM_FRONT"  ###png
# all_roots = [depth_resnet101_img_root,depth_lidar_img_root,fake_night_iccv21_img_root,fake_night_cyclegan_img_root]
# all_roots = [depth_resnet101_img_root,depth_lidar_img_root,fake_night_iccv21_img_root,fake_night_cyclegan_img_root]
# all_roots = [depth_resnet101_img_root]

# Is_png = [1,0,0,1]



#---------------------------------->  det label  path
#mats, gt_boxes, gt_labels, depth_labels
gt_boxes_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_detection/gt_boxes_f/samples/CAM_FRONT"
gt_labels_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_detection/gt_labels_f/samples/CAM_FRONT"
mats_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_detection/mats/samples/CAM_FRONT"
depth_labels_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_detection/depth_labels/samples/CAM_FRONT"



from cldm.low_dark import transfer_dark


class MyDataset(Dataset):
    def __init__(self):
        # self.condition_select = condition_select
        self.data = []
        with open(json_full_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        jpg_name = item['target']
        png_name = item['source_1']
        prompt = item['prompt']



        daytime_img = cv2.imread(os.path.join(daytime_img_root,jpg_name))
        daytime_img = cv2.resize(daytime_img, (512, 512))
        daytime_img = cv2.cvtColor(daytime_img, cv2.COLOR_BGR2RGB)
        daytime_img = (daytime_img.astype(np.float32) / 127.5) - 1.0

        conditions = []

        #  depth images condition
        condition_depth = cv2.imread(os.path.join(depth_resnet101_img_root,png_name))
        condition_depth = cv2.resize(condition_depth, (512, 512))
        condition_depth = cv2.cvtColor(condition_depth, cv2.COLOR_BGR2RGB)
        condition_depth = condition_depth.astype(np.float32) / 255.0
        conditions.append(condition_depth)


        #  faked nighttime images condition
        fake_img_condition = cv2.imread(os.path.join(daytime_img_root,jpg_name))
        condition_night  = transfer_dark(fake_img_condition) #TODO:
        condition_night = cv2.resize(condition_night, (512, 512))
        condition_night = cv2.cvtColor(condition_night, cv2.COLOR_BGR2RGB)
        condition_night = condition_night.astype(np.float32) / 255.0
        conditions.append(condition_night)


        conditions = np.concatenate(conditions, axis=2)

         

        #---------------------> det labels
        instance_name = jpg_name.split('.')[0]
        label_name = instance_name + '.pt'
        mats = torch.load(os.path.join(mats_root,label_name))
        gt_boxes = torch.load(os.path.join(gt_boxes_root,label_name))
        gt_labels = torch.load(os.path.join(gt_labels_root,label_name))
        depth_labels = torch.load(os.path.join(depth_labels_root,label_name))



        # return dict(jpg=daytime_img, txt=prompt, hint=conditions)
        return dict(jpg=daytime_img, txt=prompt, hint=conditions,mats=mats, gt_boxes=gt_boxes, gt_labels=gt_labels, depth_labels=depth_labels) 




