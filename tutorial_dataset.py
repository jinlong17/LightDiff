import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
import torch


###TODO:baolu
daytime_img_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset/train/daytime/samples/CAM_FRONT"
depth_resnet101_img_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_depth_resnet101/trainval-1.0/samples/CAM_FRONT_depth_resnet101"  ###png 
depth_lidar_img_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_lidar2depth/train/samples/CAM_FRONT"  ###jpg
fake_night_iccv21_img_root = "/home/jinlongli/personal/DATASet/Nuscene_full/1.low_dark_fixed/daytime/samples/CAM_FRONT"  ###jpg
fake_night_cyclegan_img_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_faked/train/2.cyclegan_dark/daytime/samples/CAM_FRONT"  ###png
# all_roots = [depth_resnet101_img_root,depth_lidar_img_root,fake_night_iccv21_img_root,fake_night_cyclegan_img_root]
all_roots = [depth_resnet101_img_root,depth_lidar_img_root,fake_night_iccv21_img_root,fake_night_cyclegan_img_root]


json_full_path = "/home/jinlongli/personal/DATASet/Nuscene_full/fakenightanddepth2day.json"

###simple prompt for <8> and <9> and <10>
#json_full_path = "/home/baoluli/personal/1.dataset/Nuscene_full/simple_prompt_fakenightanddepth2day.json"
Is_png = [1,0,0,1]

###TODO:det label
#mats, gt_boxes, gt_labels, depth_labels
# mats_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_detection/mats_f/samples/CAM_FRONT"
gt_boxes_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_detection/gt_boxes_f/samples/CAM_FRONT"
gt_labels_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_detection/gt_labels_f/samples/CAM_FRONT"
# depth_labels_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_detection/depth_labels_f/samples/CAM_FRONT"
mats_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_detection/mats/samples/CAM_FRONT"
# gt_boxes_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_detection/gt_boxes/samples/CAM_FRONT"
# gt_labels_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_detection/gt_labels/samples/CAM_FRONT"
depth_labels_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_detection/depth_labels/samples/CAM_FRONT"




nuscene_nighttime='/home/jinlongli/personal/DATASet/Nuscene_full/dataset/train/nighttime/samples/CAM_FRONT'

from cldm.low_dark import transfer_dark_swap_masks, transfer_dark_swap_masks_plus,transfer_daytime

class MyDataset(Dataset):
    def __init__(self,condition_select):
        self.condition_select = condition_select
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

        # prompt = item['prompt']
        #################################
        prompt = ' '





        daytime_img = cv2.imread(os.path.join(daytime_img_root,jpg_name))
        ######################TODO:jinlond add 2023.12.26
        daytime_img = transfer_daytime(daytime_img)
        ######################TODO:jinlond add 2023.12.26
        daytime_img = cv2.resize(daytime_img, (512, 512))
        daytime_img = cv2.cvtColor(daytime_img, cv2.COLOR_BGR2RGB)
        daytime_img = (daytime_img.astype(np.float32) / 127.5) - 1.0


        conditions = []
        for index in range(len(self.condition_select)):
            Is_condition = self.condition_select[index]
            if Is_condition == True:
                if index !=2:###TODO:jinlong-------index !=2
                    if Is_png[index] == True:
                        condition = cv2.imread(os.path.join(all_roots[index],png_name))
                    else:
                        condition = cv2.imread(os.path.join(all_roots[index],jpg_name))
                else:###TODO:jinlong
                        fake_img_condition = cv2.imread(os.path.join(daytime_img_root,jpg_name))

                        # condition  = transfer_dark_swap_masks(fake_img_condition, mask_size=(20, 400))
                        condition = transfer_dark_swap_masks_plus(fake_img_condition, mask_size=(20, 400), night_img_Path=nuscene_nighttime)
                
                condition = cv2.resize(condition, (512, 512))
                condition = cv2.cvtColor(condition, cv2.COLOR_BGR2RGB)
                condition = condition.astype(np.float32) / 255.0
                conditions.append(condition)
            
        conditions = np.concatenate(conditions, axis=2)
                

        ###TODO:det labels
        instance_name = jpg_name.split('.')[0]
        label_name = instance_name + '.pt'
        mats = torch.load(os.path.join(mats_root,label_name))
        gt_boxes = torch.load(os.path.join(gt_boxes_root,label_name))
        gt_labels = torch.load(os.path.join(gt_labels_root,label_name))
        depth_labels = torch.load(os.path.join(depth_labels_root,label_name))





        return dict(jpg=daytime_img, txt=prompt, hint=conditions)


        # return dict(jpg=daytime_img, txt=prompt, hint=conditions,mats=mats, gt_boxes=gt_boxes, gt_labels=gt_labels, depth_labels=depth_labels) 




