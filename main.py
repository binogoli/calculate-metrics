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
import numpy as np
from PIL import Image

import sys
sys.path.append('../')
sys.path.append('../brisque')
print(sys.path)
# from brisque import BRISQUE

# 数据路径
# 自己的数据
path_result = r'G:\LISU_Comparsion_LOL\Ours'
# ground truth
path_target = r'G:\dataset\LOLv1\eval15\high'
# path_target = r''

psnr_total, ssim_total, lpips_total, mse_total, niqe_total = 0, 0, 0, 0, 0

print('='*50, 'PSNR, SSIM, MSE and NIQE are being calculated!', '='*50, sep='\n')

image_list = os.listdir(path_result)
L = len(image_list)

psnr_total = 0
ssim_total = 0
mse_total = 0
niqe_total = 0
brisque_total = 0

for index in range(L):

    result_image_path = os.path.join(path_result, str(image_list[index]))

    # 也就是无参考时候 niqe测试时候打开
    image_result = cv2.imread(result_image_path, cv2.IMREAD_COLOR)

    # 有参考测试时候打开
    image_result = Image.open(result_image_path)
    image_result_numpy = (np.asarray(image_result) / 255.0)
    image_result = torch.from_numpy(image_result_numpy).float()

    # 四个指标
    if path_target != '':
        target_image_path = os.path.join(path_target, str(image_list[index]))
        image_target = Image.open(target_image_path)
        if image_target.size != image_result.size:
            image_target = image_target.resize((image_result.shape[1], image_result.shape[0]), Image.BILINEAR)
        image_target = (np.asarray(image_target) / 255.0)
        image_target = torch.from_numpy(image_target).float()

        # psnr_total += calculate_psnr(image_result, image_target, test_y_channel=True)
        # ssim_total += calculate_ssim(image_result, image_target, test_y_channel=True)
        ff = torch.nn.MSELoss()
        mse_total += ff(image_result, image_target)

    niqe_total += calculate_niqe(image_result_numpy, crop_border=0).item()

    print(f'\r{index + 1} / {L}', end='', flush=True)

# LPIPS计算
if path_target != '':
    print('\n' + '=' * 50, 'LPIPS is being calculated!', '=' * 50, sep='\n')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calculate_lpips = lpips.LPIPS(net='alex', verbose=False).to(device)

    datasetTest = MyDataSet(path_result, path_target)
    testLoader = DataLoader(dataset=datasetTest)

    for index, (x, y) in enumerate(testLoader):

        result, target = x.to(device), y.to(device)
        lpips_total += calculate_lpips(result * 2 - 1, target * 2 - 1).squeeze().item()

        print(f'\r{index + 1} / {L}', end='', flush=True)



print('\n' + '='*50, f'PSNR: {psnr_total / L}', f'SSIM: {ssim_total / L}', f'MSE: {mse_total / L}', f'LPIPS: {lpips_total / L}', f'NIQE: {niqe_total / L}', '='*50, f'BRISQUE: {brisque_total / L}', '='*50,sep='\n')