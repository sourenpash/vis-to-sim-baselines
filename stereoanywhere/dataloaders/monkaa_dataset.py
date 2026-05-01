from .base_dataset import BaseDataset
from glob import glob
import os.path as osp
from .frame_utils import *
import numpy as np
import os
import cv2
import torch

class MonkaaDataset(BaseDataset):
    def load_data(self, datapath):
        image2_list = sorted(glob(osp.join(datapath, 'frames_cleanpass/*/left/*.png')))
        image3_list = sorted(glob(osp.join(datapath, 'frames_cleanpass/*/right/*.png')))
        gt2_list =  sorted(glob(osp.join(datapath, 'disparity/*/left/*.pfm')))
        gt3_list =  sorted(glob(osp.join(datapath, 'disparity/*/right/*.pfm')))

        if self.mono is not None:
            image2_mono_list = sorted(glob(osp.join(datapath, f'frames_cleanpass/*/left_{self.mono}/*.png')))
            image3_mono_list = sorted(glob(osp.join(datapath, f'frames_cleanpass/*/right_{self.mono}/*.png')))

        assert len(image2_list) ==  len(image3_list) == len(gt2_list) == len(gt3_list)
        assert self.mono is None or len(image2_list) == len(image2_mono_list) == len(image3_mono_list)

        for i in range(len(image2_list)):
            _tmp = [image2_list[i], image3_list[i], gt2_list[i], gt3_list[i]]

            if self.mono is not None:
                _tmp += [image2_mono_list[i], image3_mono_list[i]]

            self.image_list += [ _tmp ]
            self.extra_info += [ image2_list[i].split('/')[-1] ] # scene and frame_id
                  
    
    def load_sample(self, index):
        data = {}

        data['im2'] = np.array(read_gen(self.image_list[index][0])).astype(np.uint8)
        data['im3'] = np.array(read_gen(self.image_list[index][1])).astype(np.uint8)

        if self.mono is not None:
            data['im2_mono'] = np.expand_dims(read_mono(self.image_list[index][4]).astype(np.float32), -1)
            data['im3_mono'] = np.expand_dims(read_mono(self.image_list[index][5]).astype(np.float32), -1)

        if self.is_test:
            data['im2'] = data['im2'] / 255.0
            data['im3'] = data['im3'] / 255.0

        # grayscale images
        data['im2'] = self.gray2rgb(data['im2'])
        data['im3'] = self.gray2rgb(data['im3'])

        data['gt'] = np.expand_dims( readPFM(self.image_list[index][2]), -1)
        data['gt'] = np.abs(data['gt'])
        data['validgt'] = (data['gt'] < 5000) & (data['gt'] > 0)

        data['gt'] = np.array(data['gt']).astype(np.float32)
        data['validgt'] = np.array(data['validgt']).astype(np.uint8)

        if os.path.exists(self.image_list[index][3]):
            data['gt_right'] = np.expand_dims( readPFM(self.image_list[index][3]), -1)
            data['gt_right'] = np.abs(data['gt_right'])
            assert np.sum(data['gt_right']>0)>0 # disabled for eth3d
        elif self.is_test:
            data['gt_right'] = np.zeros_like(data['gt'])
        else:
            raise ValueError('No right disparity available for training')
        
        data['validgt_right'] = (data['gt_right'] < 5000) & (data['gt_right'] > 0)
        data['gt_right'] = np.array(data['gt_right']).astype(np.float32)
        data['validgt_right'] = np.array(data['validgt_right']).astype(np.uint8)
        
        # data['maskocc'] = np.expand_dims(np.zeros_like(data['validgt']), -1)
        # data['maskocc'] = np.array(data['maskocc']).astype(np.uint8)

        data = self.rescale_data(data)
        
        if self.is_test or self.augmentor is None:
            data['im2_aug'] = data['im2']
            data['im3_aug'] = data['im3']
        else:
            im2_mono = data['im2_mono'] if self.mono is not None else None
            im3_mono = data['im3_mono'] if self.mono is not None else None
            data = self.augmentor(data['im2'], data['im3'], im2_mono, im3_mono, gt2=data['gt'], validgt2=data['validgt'], maskocc=None, gt3=data['gt_right'], validgt3=data['validgt_right'])

        for k in data:
            if data[k] is not None:
                data[k] = torch.from_numpy(data[k]).permute(2, 0, 1).float() 
        
        data['extra_info'] = self.extra_info[index]

        return data