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
path_result = r'.\lol'
path_target = r'G:\DeepLowLight\Zero-DCE\Zero-Dce-Semantic\对比实验结果\EnlightenGAN'
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
    image_result = cv2.imread(result_image_path, cv2.IMREAD_COLOR)

    target_image_path = os.path.join(path_target, str(image_list[index]))
    image_target = cv2.imread(target_image_path, cv2.IMREAD_COLOR)

    # 四个指标
    if path_target != '':
        psnr_total += calculate_psnr(image_result, image_target, test_y_channel=True)
        ssim_total += calculate_ssim(image_result, image_target, test_y_channel=True)
        mse_total += calculate_mse(image_result, image_target, test_y_channel=True)
    niqe_total += calculate_niqe(image_result, crop_border=0).item()

    # brisq = BRISQUE()
    # brisque_total += brisq.get_score(image_result)

    print(f'\r{index + 1} / {L}', end='', flush=True)

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