'''
Descripttion: 
version: 
Author: Jinlong Li CSU PhD
Date: 2023-11-02 15:33:06
LastEditors: Jinlong Li CSU PhD
LastEditTime: 2023-11-02 15:52:46
'''

import shutil
import pickle
import os
from nuscenes.nuscenes import NuScenes

from tqdm import tqdm

import copy

Sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',  'CAM_FRONT_LEFT']
val_path = '/home/jinlong/1.Detection_Sets/mmdetection3d/data/nuscenes/v1.0/nuscenes_infos_val.pkl'
save_path = '/home/jinlong/Desktop/TO-DO-2023'
name = 'nuscenes_infos_val_CAM_FRONT.pkl'

with open(val_path, 'rb') as file:
    loaded_data = pickle.load(file)

    # data_list是数据列表
    data_list = loaded_data['data_list']

    selected_images_info = {}

    target_file = { 'metainfo': loaded_data['metainfo'],
                   'data_list': copy.deepcopy(loaded_data['data_list'])
    }

    for i in range(len(loaded_data['data_list'])):
        info = loaded_data['data_list'][i]
        images_info = info['images']

        target_info = target_file['data_list'][i]

        for camera_name, camera_data in images_info.items():
            img_path = camera_data['img_path']

            if camera_name != 'CAM_FRONT':
                del target_info['images'][camera_name]




    target_path = os.path.join(save_path, name)

    target_folder = os.path.dirname(target_path)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)


with open(target_path, 'wb') as output_file:
    pickle.dump(target_file, output_file)

print(f"Data containing saved to {output_file}")

