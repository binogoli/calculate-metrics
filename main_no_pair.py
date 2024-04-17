import os
import cv2
import torch
import lpips
from mse import calculate_mse
from datasets import MyDataSet
from psnr import calculate_psnr
from ssim import calculate_ssim
from niqe import calculate_niqe
from torch.utils.data import DataLoader

import sys
sys.path.append('../')
sys.path.append('../brisque')
print(sys.path)
# from brisque import BRISQUE

# 数据路径
path_result = r'G:\temporaryFolder'
path_target = r''
# path_target = r''

list_folder = os.listdir(path_result)

for folder in list_folder:
    folder_path = path_result + '\\' + folder
    psnr_total, ssim_total, lpips_total, mse_total, niqe_total = 0, 0, 0, 0, 0

    image_list = os.listdir(folder_path)
    L = len(image_list)

    psnr_total = 0
    ssim_total = 0
    mse_total = 0
    niqe_total = 0
    brisque_total = 0

    for index in range(L):
        result_image_path = os.path.join(folder_path, str(image_list[index]))
        image_result = cv2.imread(result_image_path, cv2.IMREAD_COLOR)

        target_image_path = os.path.join(path_target, str(image_list[index]))
        image_target = cv2.imread(target_image_path, cv2.IMREAD_COLOR)

        # 四个指标
        niqe_total += calculate_niqe(image_result, crop_border=0).item()

    print('\n' + folder + '\n' + '='*50, f'LPIPS: {lpips_total / L}', f'NIQE: {niqe_total / L}', '='*50, '='*50,sep='\n')