from .base_dataset import BaseDataset
from glob import glob
import os.path as osp
from .frame_utils import *
import numpy as np
import os
import cv2
import torch

class FlyingThingsDataset(BaseDataset):
    def load_data(self, datapath):
        image_list = sorted(glob(osp.join(datapath, 'image_clean/left/*.png')))

        for i in range(len(image_list)):
            _tmp = [image_list[i].replace('image_clean','disparity').replace('png','pfm'),
                      image_list[i], image_list[i].replace('left', 'right'),
                      image_list[i].replace('image_clean','disparity').replace('png','pfm').replace('left', 'right'),]

            if self.mono is not None:
                _tmp += [image_list[i].replace('left', f'left_{self.mono}'), image_list[i].replace('left', f'right_{self.mono}')]
            
            self.image_list += [_tmp]
            self.extra_info += [ image_list[i].split('/')[-1] ] # scene and frame_id
    
    def load_sample(self, index):
        data = {}
        
        data['im2'] = np.array(read_gen(self.image_list[index][1])).astype(np.uint8)
        data['im3'] = np.array(read_gen(self.image_list[index][2])).astype(np.uint8)

        if self.mono is not None:
            data['im2_mono'] = np.expand_dims( read_mono(self.image_list[index][4]).astype(np.float32), -1)
            data['im3_mono'] = np.expand_dims( read_mono(self.image_list[index][5]).astype(np.float32), -1)

        if self.is_test:
            data['im2'] = data['im2'] / 255.0
            data['im3'] = data['im3'] / 255.0

        # grayscale images
        data['im2'] = self.gray2rgb(data['im2'])
        data['im3'] = self.gray2rgb(data['im3'])

        data['gt'] = np.expand_dims( -readPFM(self.image_list[index][0]), -1)
        data['validgt'] = (data['gt'] < 5000) & (data['gt'] > 0)

        data['gt'] = np.array(data['gt']).astype(np.float32)
        data['validgt'] = np.array(data['validgt']).astype(np.uint8)

        if os.path.exists(self.image_list[index][3]):
            data['gt_right'] = np.expand_dims( readPFM(self.image_list[index][3]), -1)
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