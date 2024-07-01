


import random

import torch
import numpy as np

import cv2

import scipy.stats as stats

import os
from PIL import Image

import random

from tqdm import tqdm

import torchvision.transforms as transforms


degration_cfg=dict(darkness_range=(0.01, 1.0),
                    gamma_range=(2.0, 3.5),
                    rgb_range=(0.8, 0.1),
                    red_range=(1.9, 2.4),
                    blue_range=(1.5, 1.9),
                    quantisation=[12, 14, 16]
                    )


# degration_cfg = dict(darkness_range=(0.01,0.08),  # 调整这里的范围以使图像更暗
#                     gamma_range=(0.1, 3.5),
#                     rgb_range=(1.0, 0.0),
#                     red_range=(0.8, 2.4),
#                     blue_range=(0.5, 1.9),
#                     quantisation=[12, 14, 16]
#                     )

# lower, upper = degration_cfg['darkness_range'][0], degration_cfg['darkness_range'][1]
# mu, sigma = 0.1, 0.08
# mu, sigma = 0.01, 0.008

# darkness_1 = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
# darkness_1 = darkness_1.rvs()
# print("darkness_1:  ", darkness_1)

def apply_ccm(image, ccm):
    '''
    The function of apply CCM matrix
    '''
    shape = image.shape
    image = image.reshape(-1, 3)
    image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
    return image.reshape(shape)

# def random_noise_levels():
#     """Generates random shot and read noise from a log-log linear distribution."""
#     log_min_shot_noise = np.log(0.0001)
#     log_max_shot_noise = np.log(0.012)
#     log_shot_noise = np.random.uniform(log_min_shot_noise, log_max_shot_noise)
#     shot_noise = np.exp(log_shot_noise)

#     line = lambda x: 2.18 * x + 1.20
#     log_read_noise = line(log_shot_noise) + np.random.normal(scale=0.26)
#     # print('shot noise and read noise:', log_shot_noise, log_read_noise)
#     read_noise = np.exp(log_read_noise)
#     return shot_noise, read_noise




def random_noise_levels():
    """Generates random shot and read noise from a log-log linear distribution."""

    # A_range = (0.00001, 0.0004)
    # B_range = (0.0012, 0.04)
    
    # A = random.uniform(A_range[0], A_range[1])
    # B = random.uniform(B_range[0], B_range[1])

    # log_min_shot_noise = np.log(A)
    # log_max_shot_noise = np.log(B)


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


    # mu_range = (0.01, 0.2)
    # sigma_range = (0.01, 0.2)

    # # 生成随机的 mu 和 sigma 值
    # mu = random.uniform(mu_range[0], mu_range[1])
    # sigma = random.uniform(sigma_range[0], sigma_range[1])


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
    # gamma correction
    # img9 = torch.clamp(img9, 0, 1)
    
    # img9 = img9+torch.abs(torch.min(img9))
    # print("1111111111111111111111111", torch.max(img9), torch.min(img9))
    img10 = torch.max(img9, epsilon) ** (1 / gamma)
    # print("1111111111111111111111111", torch.max(img10), torch.min(img10))



    # img_low = img10.permute(2, 0, 1)  # (H, W, C) -- (C, H, W)
    img_low = img10.permute(2, 0, 1)  # (H, W, C) -- (C, H, W)
    # img_low = torch.clamp(img_low, 0, 1)


    # degration infomations: darkness, gamma value, WB red, WB blue
    # dark_gt = torch.FloatTensor([darkness]).to(torch.device(device))
    para_gt = torch.FloatTensor([darkness, 1.0 / gamma, 1.0 / red_gain, 1.0 / blue_gain]).to(torch.device(device))
    # others_gt = torch.FloatTensor([1.0 / gamma, 1.0, 1.0]).to(torch.device(device))
    # print('the degration information:', degration_info)
    return img_low, para_gt


def generate_and_swap_masks_one(A, B, mask_size=(50, 400)):
    
    # 获取A和B的形状
    shape = A.shape

    # 生成随机大小的掩码
    mask_length = random.randint(mask_size[0], mask_size[1])
    mask_width = random.randint(mask_size[0], mask_size[1])

    # 确保掩码不超出A和B的形状
    mask_length = min(mask_length, shape[1])
    mask_width = min(mask_width, shape[2])

    # 随机选择掩码的位置
    x = random.randint(0, shape[1] - mask_length)
    y = random.randint(0, shape[2] - mask_width)

    # 创建一个与A相同形状的掩码
    mask = torch.zeros_like(A)
    mask[:, x:x + mask_length, y:y + mask_width] = 1

    # 用掩码交换A和B的内容
    A_masked = A * (1 - mask) + B * mask
    # B_masked = B * (1 - mask) + A * mask

    return A_masked




def loop_dark_generation(input_folder, output_folder):

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    i=0
    transforms_GY = transforms.ToTensor()
    transforms_BZ = transforms.Normalize(mean=[0, 0, 0], std=[255., 255., 255.])

    bar = tqdm(total=len(os.listdir(input_folder)))

    for filename in os.listdir(input_folder):
        # if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):  # 检查文件扩展名
        if filename.endswith(('.jpg')):  # 检查文件扩展名
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 打开图像并转换为Tensor
            image = Image.open(input_path)
            
            image_tensor = transforms_GY(image)

            image_tensor =(image_tensor-torch.min(image_tensor))/ (torch.max(image_tensor)-torch.min(image_tensor))

            #Low_Illumination_Degrading
            low_dark_img, gt_low_dark = Low_Illumination_Degrading(image_tensor)


            low_dark_img = torch.clamp(low_dark_img, min=0.0, max=1.0)

            tensor = (low_dark_img * 255).clamp(0, 255).byte()
            image = transforms.ToPILImage()(tensor)
            image.save(output_path)

        

        i=i+1
        bar.update(1)


    print("处理完成")
    bar.close()


import torch
import random
import copy


def generate_and_swap_masks(input_folder, output_folder):


    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    i=0
    transforms_GY = transforms.ToTensor()
    transforms_BZ = transforms.Normalize(mean=[0, 0, 0], std=[255., 255., 255.])

    bar = tqdm(total=len(os.listdir(input_folder)))

    for filename in os.listdir(input_folder):
        # if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):  # 检查文件扩展名
        if filename.endswith(('.jpg')):  # 检查文件扩展名
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            output_path_ori = os.path.join(output_folder,'1', filename)

            # 打开图像并转换为Tensor
            # image = Image.open(input_path)

            daytime_img = cv2.imread(input_path)
            image_tensor = torch.from_numpy(daytime_img)
            image_tensor = image_tensor.permute(2, 0, 1)
            
            # image_tensor = transforms_GY(image)

            image_tensor =(image_tensor-torch.min(image_tensor))/ (torch.max(image_tensor)-torch.min(image_tensor))
            image_tensor_ori = copy.deepcopy(image_tensor)
            blank_board = torch.ones_like(image_tensor)
            # blank_board = torch.zeros_like(image_tensor)



            image_tensor  = generate_and_swap_masks_one(image_tensor, blank_board, mask_size=(10, 400))
            # A ,B  = generate_and_swap_masks_two(image_tensor, blank_board)
            

            #Low_Illumination_Degrading
            low_dark_img, gt_low_dark = Low_Illumination_Degrading(image_tensor)
            # low_dark_img = image_tensor
            low_dark_img  = generate_and_swap_masks_one(low_dark_img, image_tensor_ori, mask_size=(10, 200))

            # low_dark_img =(low_dark_img-torch.min(low_dark_img))/ (torch.max(low_dark_img)-torch.min(low_dark_img))

            # low_dark_img = torch.clamp(low_dark_img, min=0.0, max=1.0)


            # A = (A * 255).clamp(0, 255).byte()
            # A = transforms.ToPILImage()(A)

            # B = (B * 255).clamp(0, 255).byte()
            # B = transforms.ToPILImage()(B)

            # A.save(output_path)
            # B.save(output_path_ori)
            
            # tensor = (low_dark_img * 255).clamp(0, 255).byte()
            tensor = (low_dark_img * 255).clamp(0, 255).numpy()
            if tensor.shape[0] == 3:
                tensor = tensor.transpose(1, 2, 0)
            cv2.imwrite(output_path, tensor)
            # image = transforms.ToPILImage()(tensor)

            # numpy_array = low_dark_img.cpu().numpy()
            # numpy_array = np.transpose(numpy_array, (1, 2, 0))
            # image = Image.fromarray((numpy_array * 255).astype(np.uint8)) 
            # image.save(output_path)

        

        i=i+1
        bar.update(1)


    print("处理完成")


###########TODO:jinlong
def add_random_light_effect(image_tensor, radius=[20,200], enhancement_factor_range=(0.6, 1)):
    """
    Add a random light effect to the image within a circular region.

    Parameters:
    - image_tensor (torch.Tensor): Input image tensor with values in the range [0, 1] and shape (3, H, W).
    - max_radius (int): Maximum radius of the circular region (default: 50).
    - enhancement_factor_range (tuple): Range for enhancement factor (default: (0.01, 0.03)).

    Returns:
    - torch.Tensor: Processed image tensor.
    """
    _, H, W = image_tensor.shape

    # Randomly select a center point
    center = (np.random.randint(0, W), np.random.randint(0, H))

    # Randomly select a radius
    radius = np.random.randint(radius[0], radius[1])

    # Create a grid of coordinates
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W))

    # Calculate the distance from each pixel to the random center
    distance = torch.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Randomly select an enhancement factor within the specified range
    enhancement_factor = np.random.uniform(*enhancement_factor_range)

    # Calculate the enhancement values based on the distance
    enhancement_values = enhancement_factor * (1 - distance / radius).clamp(0, 1)

    # Apply the enhancement to the entire image using addition
    enhanced_image = image_tensor + enhancement_values.unsqueeze(0)

    # Clip the values to ensure they remain in the valid range [0, 1]
    enhanced_image = torch.clamp(enhanced_image, 0, 1)

    return enhanced_image


def transfer_dark_swap_masks(daytime_img, mask_size):


    image_tensor = torch.from_numpy(daytime_img)
    image_tensor = image_tensor.permute(2, 0, 1)

    image_tensor =(image_tensor-torch.min(image_tensor))/ (torch.max(image_tensor)-torch.min(image_tensor))

    image_tensor_ori = copy.deepcopy(image_tensor)
    blank_board = torch.zeros_like(image_tensor)

    image_tensor  = generate_and_swap_masks_one(image_tensor, blank_board, mask_size=mask_size)
    
    #Low_Illumination_Degrading
    low_dark_img, gt_low_dark = Low_Illumination_Degrading(image_tensor)
    # low_dark_img = image_tensor
    low_dark_img  = generate_and_swap_masks_one(low_dark_img, image_tensor_ori, mask_size=mask_size)


    tensor = (low_dark_img * 255).clamp(0, 255).numpy()
    if tensor.shape[0] == 3:
        tensor = tensor.transpose(1, 2, 0)

    return tensor


###########TODO:jinlong
def normalize_and_threshold(tensor, threshold=0.3):
    """
    Normalize a tensor to the range [0, 1] and apply thresholding.

    Parameters:
    - tensor (torch.Tensor): Input tensor.
    - threshold (float): Threshold value.

    Returns:
    - torch.Tensor: Processed tensor.
    """

    # neighborhood_size=5

    # grayscale_tensor = torch.mean(tensor, dim=0)
    # local_avg_tensor = local_average(tensor, neighborhood_size)
    
    median = torch.median(tensor)
    mean = torch.mean(tensor)


    _, H, _ = tensor.shape

    top_half = tensor[:, :H//2, :]
    bottom_half = tensor[:, H//2:, :]

    threshold_top=threshold
    # threshold_bottom=0.8
    top_half_mask = top_half > threshold_top
    top_half[top_half_mask] = 0.2 * top_half[top_half_mask]

    # bottom_half_mask = bottom_half > threshold_bottom
    # bottom_half[bottom_half_mask] = 0.8 * bottom_half[bottom_half_mask]

    processed_image = torch.cat([top_half, bottom_half], dim=1)



    return processed_image



def transfer_dark_swap_masks_plus(daytime_img, mask_size=(20, 80), night_img_Path=None):
    

    image_tensor = torch.from_numpy(daytime_img)
    image_tensor = image_tensor.permute(2, 0, 1)

    image_tensor =(image_tensor-torch.min(image_tensor))/ (torch.max(image_tensor)-torch.min(image_tensor))

    image_tensor_ori = copy.deepcopy(image_tensor)
    blank_board = torch.zeros_like(image_tensor)

    # image_tensor  = generate_and_swap_masks_one(image_tensor, blank_board, mask_size=mask_size) # the second 
    
    #Low_Illumination_Degrading
    low_dark_img, gt_low_dark = Low_Illumination_Degrading(image_tensor)

    ###########TODO:jinlong
    # low_dark_img = add_random_light_effect(low_dark_img,radius=[20,200], enhancement_factor_range=(0.6, 1))
    # low_dark_img = add_random_light_effect(low_dark_img,radius=[20,200], enhancement_factor_range=(0.6, 1))
    # low_dark_img = add_random_light_effect(low_dark_img,radius=[20,200], enhancement_factor_range=(0.6, 1))
    # low_dark_img = add_random_light_effect(low_dark_img,radius=[20,200], enhancement_factor_range=(0.6, 1))
    # low_dark_img = add_random_light_effect(low_dark_img,radius=[20,200], enhancement_factor_range=(0.6, 1))


    # if night_img_Path is not None:
    #     key = random.randint(0, len(os.listdir(night_img_Path))-1)
    #     filename = os.listdir(night_img_Path)[key]
    #     input_path = os.path.join(night_img_Path, filename)
    #     nighttime_img = cv2.imread(input_path)
    #     nighttime_img = torch.from_numpy(nighttime_img)
    #     nighttime_img = nighttime_img.permute(2, 0, 1)

    #     low_dark_img  = generate_and_swap_masks_one(low_dark_img, nighttime_img, mask_size=mask_size)

    # low_dark_img  = generate_and_swap_masks_one(low_dark_img, image_tensor_ori, mask_size=mask_size)



    tensor = (low_dark_img * 255).clamp(0, 255).numpy()
    if tensor.shape[0] == 3:
        tensor = tensor.transpose(1, 2, 0)

    return tensor

import torchvision.transforms.functional as TF

def transfer_daytime(daytime_img):

    image_tensor = torch.from_numpy(daytime_img)
    image_tensor = image_tensor.permute(2, 0, 1)

    image_tensor =(image_tensor-torch.min(image_tensor))/ (torch.max(image_tensor)-torch.min(image_tensor))

    image_tensor_ori = copy.deepcopy(image_tensor)
    blank_board = torch.zeros_like(image_tensor)

    # image_tensor  = generate_and_swap_masks_one(image_tensor, blank_board, mask_size=mask_size)
    
    #Low_Illumination_Degrading
    # random_factors = np.random.uniform(3, 6, 30)################# the first 



    # random_factors = np.random.uniform(3, 4, 4)
    # factor = np.random.choice(random_factors)

    factor = 3 


    low_dark_img = TF.adjust_saturation(image_tensor, factor)


    tensor = (low_dark_img * 255).clamp(0, 255).numpy()
    if tensor.shape[0] == 3:
        tensor = tensor.transpose(1, 2, 0)

    return tensor





###########TODO:jinlong
def transfer_inference(daytime_img):
    

    image_tensor = torch.from_numpy(daytime_img)
    image_tensor = image_tensor.permute(2, 0, 1)

    image_tensor =(image_tensor-torch.min(image_tensor))/ (torch.max(image_tensor)-torch.min(image_tensor))


    tensor = normalize_and_threshold(image_tensor, threshold=0.5)


    tensor = (tensor * 255).clamp(0, 255).numpy()
    if tensor.shape[0] == 3:
        tensor = tensor.transpose(1, 2, 0)

    return tensor










if __name__=="__main__":


    # 定义输入和输出文件夹的路径
    # input_folder = '/home/jinlong/Desktop/2' #'/home/jinlong/jinlong_NAS/Dataset/Nuscene/mini/v1.0-mini/samples/CAM_BACK'  # 替换为包含输入图像的文件夹路径
    # output_folder = '/home/jinlong/Desktop/1'  # 替换为保存旋转后图像的文件夹路径

    input_folder = "/home/jinlong/Desktop/TO-DO-2023/2"
    output_folder = "/home/jinlong/Desktop/TO-DO-2023/3"
    # input_folder = "/home/jinlong/jinlong_NAS/Dataset/Nuscene/normal/Nuscene_full/dataset/train/daytime/samples/CAM_BACK"
    # output_folder = "/home/jinlong/jinlong_NAS/Dataset/Nuscene/normal/Nuscene_full/dataset_faked/train/daytime/samples/CAM_BACK"

    # loop_dark_generation(input_folder, output_folder)
    # generate_and_swap_masks(input_folder, output_folder)
