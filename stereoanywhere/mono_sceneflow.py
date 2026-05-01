# %%
import argparse
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# import sys
# sys.path.append("..")

import os.path as osp
import glob
from PIL import Image
import tqdm

import torch
import torch.nn.functional as F

import cv2
import numpy as np

from models.depth_anything_v2 import get_depth_anything_v2

# %%
device = torch.device('cuda:0')

#load monomodel and loadmonomodel using argparse
parser = argparse.ArgumentParser(description='Monocular Depth Estimation Preprocessing')
parser.add_argument('--datapath', type=str, default='', help='Path to datasets (FlyingThings3D;Monkaa;Driving)')
parser.add_argument('--monomodel', type=str, default='DAv2', help='Monocular model to use')
parser.add_argument('--loadmonomodel', type=str, default='', help='Path to monocular model')
args = parser.parse_args()

monomodel = args.monomodel
loadmonomodel = args.loadmonomodel

if monomodel == 'DAv2':
    mono_model = get_depth_anything_v2(loadmonomodel)
    mono_model = mono_model.to(device)
    mono_model.eval()

# %%

datapaths = args.datapath.split(";")

flyingthings_image_list, monkaa_image_list, driving_image_list = [], [], []

_image_list = sorted(glob.glob(osp.join(datapaths[0], 'image_clean/left/*.png')))
for i in range(len(_image_list)):
    flyingthings_image_list += [ [_image_list[i], _image_list[i].replace('left', 'right')] ]

_image2_list = sorted(glob.glob(osp.join(datapaths[1], 'frames_cleanpass/*/left/*.png')))
_image3_list = sorted(glob.glob(osp.join(datapaths[1], 'frames_cleanpass/*/right/*.png')))
for i in range(len(_image2_list)):
    monkaa_image_list += [ [_image2_list[i], _image3_list[i]] ]

_image2_list = sorted(glob.glob(osp.join(datapaths[2], 'frames_cleanpass/*/*/*/left/*.png')))
_image3_list = sorted(glob.glob(osp.join(datapaths[2], 'frames_cleanpass/*/*/*/right/*.png')))
for i in range(len(_image2_list)):
    driving_image_list += [ [_image2_list[i], _image3_list[i]] ]

image_list = flyingthings_image_list + monkaa_image_list + driving_image_list

print(len(image_list))

# %%
def normalize(x, eps=1e-4):
    if not isinstance(x, list):
        x = [x]

    prev_min = None
    prev_max = None
        
    for i in range(len(x)):
        _min = -F.max_pool2d(-x[i], (x[i].size(2), x[i].size(3)), stride=1, padding=0).detach()
        _min = _min if prev_min is None else torch.min(_min, prev_min)
        _max = F.max_pool2d(x[i], (x[i].size(2), x[i].size(3)), stride=1, padding=0).detach()
        _max = _max if prev_max is None else torch.max(_max, prev_max)
        prev_min = _min
        prev_max = _max
        
    return [(_x-_min)/(_max-_min+eps) for _x in x]

# %%

with torch.no_grad():
    for image_path in tqdm.tqdm(image_list):
        image_left = np.array(Image.open(image_path[0])).astype(np.float32) / 255.0
        image_right = np.array(Image.open(image_path[1])).astype(np.float32) / 255.0
        image_left = torch.from_numpy(image_left).permute(2, 0, 1).unsqueeze(0).float().to(device)
        image_right = torch.from_numpy(image_right).permute(2, 0, 1).unsqueeze(0).float().to(device)

        left_path = image_path[0].replace('left', f'left_{monomodel}')
        right_path = image_path[1].replace('right', f'right_{monomodel}')

        if monomodel == 'DAv2':
            mono_depths = mono_model.infer_image(torch.cat([image_left, image_right], 0), input_size_width=518, input_size_height=518)
            #Normalize depth between 0 and 1
            mono_depths = (mono_depths - mono_depths.min()) / (mono_depths.max() - mono_depths.min())
            mono_depth_left, mono_depth_right = mono_depths[0].unsqueeze(0), mono_depths[1].unsqueeze(0)
            mono_depth_left, mono_depth_right = mono_depth_left.squeeze().cpu().detach().numpy(), mono_depth_right.squeeze().cpu().detach().numpy()
        
        os.makedirs(osp.dirname(left_path), exist_ok=True)
        os.makedirs(osp.dirname(right_path), exist_ok=True)

        cv2.imwrite(left_path, np.round(mono_depth_left*65535).astype(np.uint16))
        cv2.imwrite(right_path, np.round(mono_depth_right*65535).astype(np.uint16))
