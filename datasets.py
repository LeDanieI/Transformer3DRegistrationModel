import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import glob


class OASISDataset(Dataset):
    
    def __init__(self,
                 data_path = 'data/imagesTr/OASIS_*',
                 mask_path = 'data/masksTr/OASIS_*',
                 max_trans=0.25,
                 max_angle=30,
                 device='cuda',
                 rotateonly=False):
        self.get_paths(data_path, mask_path)
        
        #
    
    def get_paths(self, data_paths, mask_path):
        
        
    def __getitem__(self, index):
        
        return
        #
    
    def __len__(self):
        return 