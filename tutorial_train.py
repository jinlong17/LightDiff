from share import *
import os 
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from omegaconf import OmegaConf

# Configs
# resume_path = '/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/models/control_sd21_ini.ckpt'
# resume_path = '/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/models/epoch=10-step=68056.ckpt'
# resume_path='/home/jinlongli/personal/DATASet/Nuscene_full/baoluli/to_jinlong/epoch=55-step=86631.ckpt'
# resume_path='/home/jinlongli/personal/personal_jinlongli/2.model_saved/cvpr2024_lighting_night/1.model_save/2023.11/3.proposed_point_5_f/lightning_logs/version_0/checkpoints/epoch=191-step=56640.ckpt'
# resume_path='/home/jinlongli/personal/2.model_saved/cvpr2024_lighting_night/1.model_save/2024.01/3.proposed_point_5_f_day_locked/lightning_logs/version_0/checkpoints/epoch=324-step=502775.ckpt'
batch_size = 10
logger_freq = 300 #300
learning_rate = 1e-5 #1e-5
sd_locked = False #False #True
only_mid_control = False
max_epochs=2500
num_workers=16
every_n_epochs=4
# model_name='det_0.001_only_nomasks'
# model_name='det_0.0001_depth_0.1_mmd_100'
default_root_dir= './1.model_save'
# model_name='2.fixed_proposed_depth_mmd_att_lidar'
# model_name='3.proposed_point_5_f_unlocked'
# model_name='3.proposed_point_5_f_day_locked'
# model_name='3.proposed_point_5_f_day_unlocked_f'




################################## CVPR Rebuttal 

# model_name='0.img_depth'
# resume_path='/home/jinlongli/personal/2.model_saved/cvpr2024_lighting_night/2.rebuttal/for_jinlong/img_depth/epoch=10-step=17016.ckpt'
# yaml_config='/home/jinlongli/personal/2.model_saved/cvpr2024_lighting_night/2.rebuttal/for_jinlong/img_depth/rebuttal_2.yaml'

# model_name='0.img_text'
# resume_path='/home/jinlongli/personal/2.model_saved/cvpr2024_lighting_night/2.rebuttal/for_jinlong/img_text/epoch=10-step=17016.ckpt'
# yaml_config='/home/jinlongli/personal/2.model_saved/cvpr2024_lighting_night/2.rebuttal/for_jinlong/img_text/rebuttal_3.yaml'

# model_name='0.img_only'
# resume_path='/home/jinlongli/personal/2.model_saved/cvpr2024_lighting_night/2.rebuttal/for_jinlong/only_img/epoch=10-step=17016.ckpt'
# yaml_config='/home/jinlongli/personal/2.model_saved/cvpr2024_lighting_night/2.rebuttal/for_jinlong/only_img/rebuttal_1.yaml'

################################## CVPR Rebuttal 




model_name='4.proposed_day_unlocked_f_2'
# resume_path='/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/1.model_save/3.proposed_point_5_f_day_unlocked_f/lightning_logs/version_0/checkpoints/epoch=79-step=123760.ckpt'
resume_path='/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/1.model_save/4.proposed_day_unlocked_f_1/lightning_logs/version_0/checkpoints/epoch=21-step=10890.ckpt'
yaml_config='/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/models/<11>.yaml'





condition_select = OmegaConf.load(yaml_config).condition['condition_select']
model = create_model(yaml_config).cpu()


target_path = os.path.join(default_root_dir, model_name)
if not os.path.exists(target_path):
    os.makedirs(target_path)
# img_save_path = os.path.join(target_path, '0.img_log')
# if not os.path.exists(img_save_path):
#     os.makedirs(img_save_path)
default_root_dir = target_path




# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
#model = create_model('/home/baoluli/1.code/3.ControlNet/models/cldm_v21.yaml').cpu()
# model = create_model('/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/models/<11>.yaml').cpu()
# model = create_model('/home/jinlongli/personal/DATASet/Nuscene_full/baoluli/to_jinlong/<12>.yaml').cpu()

model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control





# Misc

# depth_resnet101_img_root = "/home/baoluli/personal/1.dataset/Nuscene_full/dataset_depth_resnet101/trainval-1.0/samples/CAM_FRONT_depth_resnet101"  ###png 
# depth_lidar_img_root = "/home/baoluli/personal/1.dataset/Nuscene_full/dataset_lidar2depth/train/samples/CAM_FRONT"  ###jpg
# fake_night_iccv21_img_root = "/home/baoluli/personal/1.dataset/Nuscene_full/dataset_faked/train/1.low_dark_fixed/daytime/samples/CAM_FRONT"  ###jpg
# fake_night_cyclegan_img_root = "/home/baoluli/personal/1.dataset/Nuscene_full/dataset_faked/train/2.cyclegan_dark/daytime/samples/CAM_FRONT"  ###png
# condition_select = OmegaConf.load('/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/models/<11>.yaml').condition['condition_select']

dataset = MyDataset(condition_select)
dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
# trainer = pl.Trainer(gpus=[0], precision=32, callbacks=[logger])

# trainer = pl.Trainer(gpus=[gpus], precision=32, callbacks=[logger], default_root_dir=default_root_dir, max_epochs=max_epochs, callbacks=[pl.callbacks.ModelCheckpoint(every_n_epochs=1)])
trainer = pl.Trainer(strategy='ddp',gpus=[7,6,5,4,3], precision=32, callbacks=[pl.callbacks.ModelCheckpoint(every_n_epochs=every_n_epochs, save_top_k=-1), logger], default_root_dir=default_root_dir, max_epochs=max_epochs)


# Train!
trainer.fit(model, dataloader)


