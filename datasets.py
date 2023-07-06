import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from glob import glob
from utils import set_seed
from model.RegistrationNetworks import euler_angles_to_matrix
import os
import SimpleITK as sitk
import torch.nn.functional as F


#import pdb

class OASISDataset(Dataset):
    
    def __init__(self,
                 data_path = 'data/imagesTr/OASIS_*',
                 mask_path = 'data/masksTr/OASIS_*',
                 max_trans=0.25,
                 max_angle=30,
                 device='cuda',
                 rotateonly=False):
        self.data_paths, self.mask_paths = self.get_paths(data_path, mask_path)
        self.generate_T(max_trans, max_angle)
        #
    
    def get_paths(self, data_path, mask_path):
        data_paths = glob(data_path)
        if len(data_paths)==0:
            raise Exception("Data not found. Check image/mask path")
        data_paths.sort()
        mask_paths = glob(mask_path)
        mask_paths.sort()
        return data_paths, mask_paths    
    
    def generate_T(self, max_trans, max_angle):
        self.T_real = []
        self.T_inv = []
        for i in range(len(self.data_paths)):
            set_seed(i)
            rand_trans = np.random.uniform(low=-max_trans, high=max_trans, size=(3,)).astype('float32')
            rand_angles = np.random.uniform(low=-max_angle, high=max_angle, size=(3,)).astype('float32')
            translation = torch.from_numpy(rand_trans)
            euler_angles = np.pi * torch.from_numpy(rand_angles) / 180.
            rot_mat = euler_angles_to_matrix(euler_angles=euler_angles, convention="XYZ")
            T = torch.cat((rot_mat, translation.view(3, 1)), axis=1)
            T = T.view(-1, 3, 4)
            T4x4 = torch.cat((T.squeeze(), torch.Tensor([0,0,0,1]).unsqueeze(0)),0)
            Tinv=torch.inverse(T4x4)   
            Tinv=Tinv[:-1]
            self.T_real.append(T)
            self.T_inv.append(Tinv)
    
    def read_image_np(self, path):
        if os.path.exists(path):
            image = sitk.ReadImage(path)
            image_np = sitk.GetArrayFromImage(image).astype('float32')
        else:
            print('Image does not exist')
        return image_np
    
    def transform_rigid(self, T, input_tensor):
        grid = F.affine_grid(T, input_tensor.size(),align_corners=False) #N*3*4 & N*C*D*H*W = 1,1,192,224,160
        input_aug_tensor = F.grid_sample(input_tensor, grid,align_corners=False).squeeze(0)   
        return input_aug_tensor       
    
    def augmentation(self, idx):
        fixed_img_np = self.read_image_np(self.data_paths[idx])
        fixed_img = torch.from_numpy(fixed_img_np).unsqueeze(0)
        moving_img = self.transform_rigid(self.T_real[idx],fixed_img.unsqueeze(0)) 
        fixed_mask_np = self.read_image_np(self.mask_paths[idx])
        fixed_mask = torch.from_numpy(fixed_mask_np).unsqueeze(0)
        moving_mask = self.transform_rigid(self.T_real[idx],fixed_mask.unsqueeze(0))
        moving_mask = torch.where(moving_mask < 0.5, torch.zeros_like(moving_mask), torch.ones_like(moving_mask))
        return moving_img, fixed_img, moving_mask, fixed_mask

        
    def __getitem__(self, index):       
        moving_img, fixed_img, moving_mask, fixed_mask = self.augmentation(index)
        return moving_img[0], fixed_img[0], moving_mask[0], fixed_mask[0]
        #
    
    def __len__(self):
        return 