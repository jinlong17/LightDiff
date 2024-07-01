from share import *
import config

import cv2
import einops
#import gradio as gr
import numpy as np
import torch
import random
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "5" 
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.midas import MidasDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import json
from omegaconf import OmegaConf

config_path = '/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/models/2.fixed_proposed_depth_mmd_att_lidar.yaml'
# model_path = '/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/1.model_save/proposed_wo_loss_add_nightmask/lightning_logs/version_0/checkpoints/epoch=11-step=21215.ckpt'


# model_path='/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/1.model_save/det_0.001_depth_mmd_add_nightmask/lightning_logs/version_0/checkpoints/epoch=1-step=49489.ckpt'

# model_path='/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/1.model_save/det_0.001_depth_mmd/lightning_logs/version_0/checkpoints/epoch=1-step=49489.ckpt'


# model_path='/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/1.model_save/proposed_nomasks/lightning_logs/version_0/checkpoints/epoch=19-step=35359.ckpt'
# model_path='/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/1.model_save/proposed_wo_loss_add_nightmask/lightning_logs/version_0/checkpoints/epoch=31-step=56575.ckpt'
model_path='/home/jinlongli/personal/2.model_saved/cvpr2024_lighting_night/1.model_save/proposed_nomask_sd_unlocked/lightning_logs/version_1/checkpoints/epoch=5-step=12377.ckpt'

# model_path='/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/1.model_save/1.nomask_proposed_det_0.001/lightning_logs/version_0/checkpoints/epoch=3-step=98979.ckpt'
#model_path='/home/jinlongli/personal/2.model_saved/cvpr2024_lighting_night/1.model_save/2.fixed_proposed_depth_mmd_att_lidar/lightning_logs/version_0/checkpoints/epoch=0-step=24744.ckpt'





# model_path='/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/1.model_save/2.fixed_proposed_depth_mmd_att_lidar/lightning_logs/version_0/checkpoints/epoch=2-step=74234.ckpt'
# model_path='/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/1.model_save/1.nomask_proposed_mmd_depth/lightning_logs/version_0/checkpoints/epoch=2-step=74234.ckpt'
# model_path='/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/1.model_save/proposed_wo_loss_add_nightmask/lightning_logs/version_0/checkpoints/epoch=31-step=56575.ckpt'

#model_path = '/home/jinlongli/personal/2.model_saved/cvpr2024_lighting_night/1.model_save/2.fixed_proposed_depth_mmd_att_lidar/lightning_logs/version_0/checkpoints/epoch=4-step=123724.ckpt'

# config_path = '/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/models/2.fixed_proposed_depth_mmd_att_lidar.yaml'
# model_path='/home/jinlongli/personal/2.model_saved/cvpr2024_lighting_night/1.model_save/2.fixed_proposed_depth_mmd_att_lidar/lightning_logs/version_0/checkpoints/epoch=2-step=74234.ckpt'


# config_path = '/home/jinlongli/personal/2.model_saved/cvpr2024_lighting_night/2.rebuttal/for_jinlong/only_img/rebuttal_1.yaml'
# model_path='/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/1.model_save/0.img_only/lightning_logs/version_0/checkpoints/epoch=5-step=148470.ckpt'

# config_path = '/home/jinlongli/personal/2.model_saved/cvpr2024_lighting_night/2.rebuttal/for_jinlong/img_depth/rebuttal_2.yaml'
# #model_path='/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/1.model_save/0.img_depth/lightning_logs/version_0/checkpoints/epoch=5-step=148470.ckpt'
# model_path='/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/1.model_save/0.img_depth/lightning_logs/version_0/checkpoints/epoch=7-step=197960.ckpt'

# config_path = '/home/jinlongli/personal/2.model_saved/cvpr2024_lighting_night/2.rebuttal/for_jinlong/img_text/rebuttal_3.yaml'
# model_path='/home/jinlongli/1.Detection_Set/Dark_Diffusion/ControlNet/1.model_save/0.img_text/lightning_logs/version_0/checkpoints/epoch=5-step=148470.ckpt'

condition_select = OmegaConf.load(config_path).condition['condition_select']



model = create_model(config_path).cpu()
model.load_state_dict(load_state_dict(model_path, location='cuda'))#,strict=False
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        control = input_image

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        H,W = image_resolution
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results 




# camera_list = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
camera_list = ['CAM_FRONT']


for camera_name in camera_list:
    Is_png = [1,0,0,0]
    depth_resnet101_img_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_depth_resnet101/trainval-1.0/samples/"+camera_name+"_depth_resnet101"  ###png 
    depth_lidar_img_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_lidar2depth/val/samples/"+camera_name  ###jpg
    fake_night_iccv21_img_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset/val/nighttime/samples/"+camera_name  ###jpg
    fake_night_cyclegan_img_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset/val/nighttime/samples/"+camera_name  ###jpg
    all_roots = [depth_resnet101_img_root,depth_lidar_img_root,fake_night_iccv21_img_root,fake_night_cyclegan_img_root]


    save_path = '/home/jinlongli/personal/DATASet/Nuscene_full/baoluli/0.cvpr_rebuttal/proposed_nomask_sd_unlocked_img_depth_0.8_0_5/'+camera_name####################



    if not os.path.exists(save_path):
    # 如果路径不存在，使用os.makedirs创建路径
        os.makedirs(save_path)

    json_path = '/home/jinlongli/personal/DATASet/Nuscene_full/val_night_'+camera_name+'.json'


    #1
    strength = 0.8
    scale = 0.

    #2
    # strength = 1.2
    # scale = 0.

    # strength = 1.2
    # scale = 12.


    #3
    ddim_steps = 5

    #prompt = ''#'a white truck driving down a street next to a tall building'
    a_prompt = 'best quality, extremely detailed, realistic style, daytime traffic scene, rich true color levels, pastel tones'
    n_prompt = 'lots of noise, overexposure,deformity, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    num_samples = 1
    image_resolution = (512,512) 
    detect_resolution = (512,512)
    guess_mode = False
    seed = 1000
    eta = 0


    #1 depth 
    #2 night
    #3 depth + night
    #inference_num = 3 

    #index = 0
    with open(json_path, 'rt') as f:
        for line in f:
            data = json.loads(line)
        
            jpg_name = data['target']
            png_name = data['source_1']
            #prompt = data['prompt']
            prompt = ''


            save_full_path = os.path.join(save_path,jpg_name)
            if os.path.exists(save_full_path):

                print('the generated image is exists in  ', save_full_path)

                continue

            rebuttal_condition_select = [1,1,1,0]
            conditions = []
            for index in range(len(condition_select)):
                Is_condition = condition_select[index]
                if Is_condition == True:
                    if Is_png[index] == True:
                        condition = cv2.imread(os.path.join(all_roots[index],png_name))
                    else:
                        condition = cv2.imread(os.path.join(all_roots[index],jpg_name))

                    condition = cv2.resize(condition, (512, 512))
                    if rebuttal_condition_select[index] == 0:
                        condition = np.zeros_like(condition)
                    condition = cv2.cvtColor(condition, cv2.COLOR_BGR2RGB)
                    condition = condition.astype(np.float32) / 255.0
                    conditions.append(condition)
            conditions = np.concatenate(conditions, axis=2)



            input_image = conditions
            input_image = torch.from_numpy(input_image.copy()).float().cuda() #/ 255.0
            input_image = torch.stack([input_image for _ in range(num_samples)], dim=0)
            input_image = einops.rearrange(input_image, 'b h w c -> b c h w').clone()

            # n = 10
            # strength_list = np.linspace(0,2,n)
            # scale_list = np.linspace(0,30,n)

            # horizontally_list = []
            # for i in range(n):
            #     vertically_list = []
            #     for j in range(n):
            #         strength = strength_list[i]
            #         scale = scale_list[j]
            #         ret = process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
            #         vertically_list.append(ret[0])
            #     vertically_stacked = np.vstack(vertically_list)
            #     horizontally_list.append(vertically_stacked)
            # final_img = np.hstack(horizontally_list)

            # strength = 0.8
            # scale = 3.
            ret = process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
            final_img = ret[0]

            
            # save_full_path = os.path.join(save_path,jpg_name)
            cv2.imwrite(save_full_path, final_img)
            print(jpg_name)

