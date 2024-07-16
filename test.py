from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random
import os 
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.midas import MidasDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import json
from omegaconf import OmegaConf


from cldm.low_dark import transfer_dark_swap_masks, transfer_dark_swap_masks_plus,transfer_inference


#--------------------------------------------> Configs
config_path='./models/lightdiff_v15.yaml'
model_path='/checkpoints'
name='3.proposed_point_5_f_day_locked_104_sample_one_1'

strength =1 #0.8

scale = 3

ddim_steps =50
camera_name='CAM_FRONT'
# Is_png = [1,0,0,0]
depth_resnet101_img_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset_depth_resnet101/trainval-1.0/samples/CAM_FRONT_depth_resnet101"  ###png 
night_img_root = "/home/jinlongli/personal/DATASet/Nuscene_full/dataset/val/nighttime/samples/"+camera_name  ###jpg


save_file='/home/jinlongli/personal/2.model_saved/cvpr2024_lighting_night/new_log'
save_path = os.path.join(save_file,name, camera_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

json_path = '/home/jinlongli/personal/DATASet/Nuscene_full/val_night_'+camera_name+'.json'

a_prompt = 'best quality, extremely detailed, realistic style, daytime traffic scene, rich true color levels, pastel tones'
n_prompt = 'lots of noise, overexposure,deformity, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
num_samples = 1
image_resolution = (512,512)
detect_resolution = (512,512)
guess_mode = False
seed = 1000
eta = 0

#--------------------------------------------> Configs

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


with open(json_path, 'rt') as f:
    for line in f:
        data = json.loads(line)
    
        jpg_name = data['target']
        png_name = data['source_1']
        prompt = data['prompt']
        # prompt = 'best quality, extremely detailed, realistic style, daytime traffic scene, rich true color levels, pastel tones'

        save_full_path = os.path.join(save_path,jpg_name)
        if os.path.exists(save_full_path):

            print('the generated image is exists in  ', save_full_path)

            continue

        conditions = []
        
        #  depth images condition
        condition_depth = cv2.imread(os.path.join(depth_resnet101_img_root,png_name))
        condition_depth = cv2.resize(condition_depth, (512, 512))
        condition_depth = cv2.cvtColor(condition_depth, cv2.COLOR_BGR2RGB)
        condition_depth = condition_depth.astype(np.float32) / 255.0
        conditions.append(condition_depth)

        #  faked nighttime images condition
        night_img_condition = cv2.imread(os.path.join(night_img_root,jpg_name))
        condition_night = cv2.resize(night_img_condition, (512, 512))
        condition_night = cv2.cvtColor(condition_night, cv2.COLOR_BGR2RGB)
        condition_night = condition_night.astype(np.float32) / 255.0
        conditions.append(condition_night)

        conditions = np.concatenate(conditions, axis=2)

        input_image = conditions
        input_image = torch.from_numpy(input_image.copy()).float().cuda() #/ 255.0
        input_image = torch.stack([input_image for _ in range(num_samples)], dim=0)
        input_image = einops.rearrange(input_image, 'b h w c -> b c h w').clone()



        ret = process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
        final_img = ret[0]

        
        # save_full_path = os.path.join(save_path,jpg_name)
        cv2.imwrite(save_full_path, final_img)
        print(jpg_name)

