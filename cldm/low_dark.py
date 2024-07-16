


import random

import torch
import numpy as np
import cv2
import scipy.stats as stats

import os
from PIL import Image
import random

import torchvision.transforms as transforms
from tqdm import tqdm


degration_cfg=dict(darkness_range=(0.01, 1.0),
                    gamma_range=(2.0, 3.5),
                    rgb_range=(0.8, 0.1),
                    red_range=(1.9, 2.4),
                    blue_range=(1.5, 1.9),
                    quantisation=[12, 14, 16]
                    )



def apply_ccm(image, ccm):
    '''
    The function of apply CCM matrix
    '''
    shape = image.shape
    image = image.reshape(-1, 3)
    image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
    return image.reshape(shape)



def random_noise_levels():
    """Generates random shot and read noise from a log-log linear distribution."""


    log_min_shot_noise = np.log(0.0001)
    log_max_shot_noise = np.log(0.012)

    log_shot_noise = np.random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = np.exp(log_shot_noise)

    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + np.random.normal(scale=0.26)
    # print('shot noise and read noise:', log_shot_noise, log_read_noise)
    read_noise = np.exp(log_read_noise)
    return shot_noise, read_noise




def Low_Illumination_Degrading(img, safe_invert=True):

    '''
    (1)unprocess part(RGB2RAW) (2)low light corruption part (3)ISP part(RAW2RGB)
    Some code copy from 'https://github.com/timothybrooks/unprocessing', thx to their work ~
    input:
    img (Tensor): Input normal light images of shape (C, H, W).
    img_meta(dict): A image info dict contain some information like name ,shape ...
    return:
    img_deg (Tensor): Output degration low light images of shape (C, H, W).
    degration_info(Tensor): Output degration paramter in the whole process.
    '''

    '''
    parameter setting
    '''
    device = img.device
    config = degration_cfg
    # camera color matrix
    xyz2cams = [[[1.0234, -0.2969, -0.2266],
                    [-0.5625, 1.6328, -0.0469],
                    [-0.0703, 0.2188, 0.6406]],
                [[0.4913, -0.0541, -0.0202],
                    [-0.613, 1.3513, 0.2906],
                    [-0.1564, 0.2151, 0.7183]],
                [[0.838, -0.263, -0.0639],
                    [-0.2887, 1.0725, 0.2496],
                    [-0.0627, 0.1427, 0.5438]],
                [[0.6596, -0.2079, -0.0562],
                    [-0.4782, 1.3016, 0.1933],
                    [-0.097, 0.1581, 0.5181]]]
    rgb2xyz = [[0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041]]

    # noise parameters and quantization step

    '''
    (1)unprocess part(RGB2RAW): 1.inverse tone, 2.inverse gamma, 3.sRGB2cRGB, 4.inverse WB digital gains
    '''
    img1 = img.permute(1, 2, 0)  # (C, H, W) -- (H, W, C)
    # print(img1.shape)
    # img_meta = img_metas[i]
    # inverse tone mapping
    img1 = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * img1) / 3.0)
    # inverse gamma
    epsilon = torch.FloatTensor([1e-8]).to(torch.device(device))
    gamma = random.uniform(config['gamma_range'][0], config['gamma_range'][1])
    img2 = torch.max(img1, epsilon) ** gamma
    # sRGB2cRGB
    xyz2cam = random.choice(xyz2cams)
    rgb2cam = np.matmul(xyz2cam, rgb2xyz)
    rgb2cam = torch.from_numpy(rgb2cam / np.sum(rgb2cam, axis=-1)).to(torch.float).to(torch.device(device))
    # print(rgb2cam)
    img3 = apply_ccm(img2, rgb2cam)
    img3 = torch.clamp(img3, min=0.0, max=1.0)

    # inverse WB
    rgb_gain = random.normalvariate(config['rgb_range'][0], config['rgb_range'][1])
    red_gain = random.uniform(config['red_range'][0], config['red_range'][1])
    blue_gain = random.uniform(config['blue_range'][0], config['blue_range'][1])

    gains1 = np.stack([1.0 / red_gain, 1.0, 1.0 / blue_gain]) * rgb_gain
    # gains1 = np.stack([1.0 / red_gain, 1.0, 1.0 / blue_gain])
    gains1 = gains1[np.newaxis, np.newaxis, :]
    gains1 = torch.FloatTensor(gains1).to(torch.device(device))
    # safe_invert = True
    # color disorder !!!
    if safe_invert:
        img3_gray = torch.mean(img3, dim=-1, keepdim=True)
        inflection = 0.9
        zero = torch.zeros_like(img3_gray).to(torch.device(device))
        mask = (torch.max(img3_gray - inflection, zero) / (1.0 - inflection)) ** 2.0
        safe_gains = torch.max(mask + (1.0 - mask) * gains1, gains1)

        img4 = img3 * gains1
        # img4 = torch.clamp(img3*safe_gains, min=0.0, max=1.0)

    else:
        img4 = img3 * gains1

    '''
    (2)low light corruption part: 5.darkness, 6.shot and read noise 
    '''
    # darkness(low photon numbers)
    lower, upper = config['darkness_range'][0], config['darkness_range'][1]


    mu, sigma = 0.1, 0.08
    darkness = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    # darkness = darkness_1
    darkness = darkness.rvs()
    # print(darkness)
    img5 = img4 * darkness
    # img5 = img4*0.
    # add shot and read noise
    shot_noise, read_noise = random_noise_levels()
    var = img5 * shot_noise + read_noise  # here the read noise is i    # var = img5 + read_noise # here the read noise is independent
    # print('the var is:', var)
    var = torch.max(var, epsilon)
    
    noise = torch.normal(mean=0, std=torch.sqrt(var))

    img6 = img5 + noise

    '''
    (3)ISP part(RAW2RGB): 7.quantisation  8.white balance 9.cRGB2sRGB 10.gamma correction
    '''
    # quantisation noise: uniform distribution
    bits = random.choice(config['quantisation'])
    quan_noise = torch.FloatTensor(img6.size()).uniform_(-1 / (255 * bits), 1 / (255 * bits)).to(
        torch.device(device))
    # print(quan_noise)
    # img7 = torch.clamp(img6 + quan_noise, min=0)
    img7 = img6 + quan_noise
    # white balance
    gains2 = np.stack([red_gain, 1.0, blue_gain])
    gains2 = gains2[np.newaxis, np.newaxis, :]
    gains2 = torch.FloatTensor(gains2).to(torch.device(device))
    img8 = img7 * gains2
    # img8 = img7
    # cRGB2sRGB
    cam2rgb = torch.inverse(rgb2cam)
    img9 = apply_ccm(img8, cam2rgb)

    img10 = torch.max(img9, epsilon) ** (1 / gamma)




    # img_low = img10.permute(2, 0, 1)  # (H, W, C) -- (C, H, W)
    img_low = img10.permute(2, 0, 1)  # (H, W, C) -- (C, H, W)



    # degration infomations: darkness, gamma value, WB red, WB blue
    # dark_gt = torch.FloatTensor([darkness]).to(torch.device(device))
    para_gt = torch.FloatTensor([darkness, 1.0 / gamma, 1.0 / red_gain, 1.0 / blue_gain]).to(torch.device(device))

    return img_low, para_gt





def transfer_dark(daytime_img):


    image_tensor = torch.from_numpy(daytime_img)
    image_tensor = image_tensor.permute(2, 0, 1)

    image_tensor =(image_tensor-torch.min(image_tensor))/ (torch.max(image_tensor)-torch.min(image_tensor))


    #Low_Illumination_Degrading
    low_dark_img, gt_low_dark = Low_Illumination_Degrading(image_tensor)

    tensor = (low_dark_img * 255).clamp(0, 255).numpy()
    if tensor.shape[0] == 3:
        tensor = tensor.transpose(1, 2, 0)

    return tensor




def loop_dark_generation(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    i=0
    transforms_GY = transforms.ToTensor()

    bar = tqdm(total=len(os.listdir(input_folder)))

    for filename in os.listdir(input_folder):
        # if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):  
        if filename.endswith(('.jpg')):  
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            # Tensor
            image = Image.open(input_path)
            
            image_tensor = transforms_GY(image)

            image_tensor =(image_tensor-torch.min(image_tensor))/ (torch.max(image_tensor)-torch.min(image_tensor))

            #Low_Illumination_Degrading
            low_dark_img, gt_low_dark = Low_Illumination_Degrading(image_tensor)

            tensor = (low_dark_img * 255).clamp(0, 255).byte()
            image = transforms.ToPILImage()(tensor)
            image.save(output_path)

        i=i+1
        bar.update(1)

    print("done!!!!!!!")
    bar.close()







if __name__=="__main__":


    input_folder = "/2"
    output_folder = "/3"

    loop_dark_generation(input_folder, output_folder)

