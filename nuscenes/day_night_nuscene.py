

import shutil
import pickle
import os
from nuscenes.nuscenes import NuScenes

from tqdm import tqdm

# nuscene_path='/home/jinlong/jinlong_NAS/Dataset/Nuscene/mini/v1.0-mini'

# nusc = NuScenes(version='v1.0-mini', dataroot=nuscene_path, verbose=True)




nuscene_path='/home/jinlong/1.Detection_Sets/mmdetection3d/data/nuscenes'

nusc = NuScenes(version='v1.0-test', dataroot=nuscene_path, verbose=True)





nusc.list_scenes()

# Sensors = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'LIDAR_TOP', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',  'CAM_FRONT_LEFT']

Sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',  'CAM_FRONT_LEFT']

# my_sample = nusc.sample[10]
# nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP')


### the path to store the dataset information
# train_path = '/home/jinlong/jinlong_NAS/Dataset/Nuscene/mini/v1.0-mini/nuscenes_infos_train.pkl'
# val_path = '/home/jinlong/jinlong_NAS/Dataset/Nuscene/mini/v1.0-mini/nuscenes_infos_val.pkl'
# test_path = '/home/jinlong/jinlong_NAS/Dataset/Nuscene/mini/v1.0-mini/nuscenes_infos_val.pkl'

train_path = '/home/jinlong/1.Detection_Sets/mmdetection3d/data/nuscenes/nuscenes_infos_train.pkl'
val_path = '/home/jinlong/1.Detection_Sets/mmdetection3d/data/nuscenes/nuscenes_infos_val.pkl'
test_path = '/home/jinlong/jinlong_NAS/Dataset/Nuscene/normal/v1.0-test_blobs/nuscenes_infos_test.pkl'


###save to the path 
folder_train = '/home/jinlong/jinlong_NAS/Dataset/Nuscene/normal/Nuscene_full/dataset/train'
folder_val =  '/home/jinlong/jinlong_NAS/Dataset/Nuscene/normal/Nuscene_full/dataset/val'
folder_test = '/home/jinlong/jinlong_NAS/Dataset/Nuscene/normal/Nuscene_full/dataset/test'


# folder_a_root = '/home/jinlong/jinlong_NAS/Dataset/Nuscene/mini/v1.0-mini'
folder_a_root = '/home/jinlong/1.Detection_Sets/mmdetection3d/data/nuscenes'



train_dataset = []
val_dataset = []
test_dataset = []




Sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',  'CAM_FRONT_LEFT']
test_path = '/home/jinlong/jinlong_NAS/Dataset/Nuscene/normal/v1.0-test_blobs/nuscenes_infos_test.pkl'
with open(test_path, 'rb') as file:
    loaded_data = pickle.load(file)

    # data_list
    data_list = loaded_data['data_list']

    # info
    for info in data_list:
        images_info = info['images']

        # 
        for camera_name, camera_data in images_info.items():
            img_path = camera_data['img_path']

            source_path = os.path.join('samples', camera_name, img_path)
            test_dataset.append(source_path)



# scenes = [nusc.scene[-1]]

val_dataset = []

train_dataset = []

scenes = nusc.scene

for each_scene in scenes:

    first_sample_token = each_scene['first_sample_token']
    last_sample_token  = each_scene['last_sample_token']
    num_samples = each_scene['nbr_samples']
    description = each_scene['description']
    next_token = first_sample_token
    num_count = 0


    if description.find('Night') != -1:

        print(each_scene['name'], " have 'night'")

        while (num_count!=(num_samples+1)):

            my_sample = nusc.get('sample', next_token)

            if next_token != last_sample_token:

                next_token = my_sample['next']
            else:

                print('end! ', last_sample_token,  '  ', next_token)

            for cam in Sensors:

                cam_data = nusc.get('sample_data', my_sample['data'][cam])

                cam_path = cam_data['filename']
                source_path = os.path.join(folder_a_root, cam_path)


                if cam_path in train_dataset:


                    target_path = os.path.join(folder_train, 'nighttime', cam_path)

                    target_folder = os.path.dirname(target_path)
                    if not os.path.exists(target_folder):
                        os.makedirs(target_folder)

                    shutil.copyfile(source_path, target_path)
                
                elif cam_path in val_dataset:


                    target_path = os.path.join(folder_val, 'nighttime', cam_path)

                    target_folder = os.path.dirname(target_path)
                    if not os.path.exists(target_folder):
                        os.makedirs(target_folder)
                    shutil.copyfile(source_path, target_path)

                elif cam_path in test_dataset:
    

                    target_path = os.path.join(folder_test, 'nighttime', cam_path)

                    target_folder = os.path.dirname(target_path)
                    if not os.path.exists(target_folder):
                        os.makedirs(target_folder)
                    shutil.copyfile(source_path, target_path)


                else:


                    print('None ', cam_path)


                # print('night', cam_path)

            num_count = num_count + 1

    else:

        print(each_scene['name'], "  do not have 'night'")

        while (num_count!=(num_samples+1)):
    
            my_sample = nusc.get('sample', next_token)

            if next_token != last_sample_token:

                next_token = my_sample['next']
            else:

                print('end! ', last_sample_token,  '  ', next_token)

            for cam in Sensors:

                cam_data = nusc.get('sample_data', my_sample['data'][cam])
                cam_path = cam_data['filename']


                source_path = os.path.join(folder_a_root, cam_path)


                if cam_path in train_dataset:

                    target_path = os.path.join(folder_train, 'daytime', cam_path)

                    target_folder = os.path.dirname(target_path)
                    if not os.path.exists(target_folder):
                        os.makedirs(target_folder)

                    shutil.copyfile(source_path, target_path)

                elif cam_path in val_dataset:


                    target_path = os.path.join(folder_val, 'daytime', cam_path)

                    target_folder = os.path.dirname(target_path)
                    if not os.path.exists(target_folder):
                        os.makedirs(target_folder)

                    shutil.copyfile(source_path, target_path)

                elif cam_path in test_dataset:
    

                    target_path = os.path.join(folder_test, 'daytime', cam_path)

                    target_folder = os.path.dirname(target_path)
                    if not os.path.exists(target_folder):
                        os.makedirs(target_folder)

                    shutil.copyfile(source_path, target_path)

                    
                else:

                    print('None ', cam_path)


                # print('daytime ', cam_path)

            num_count = num_count + 1
