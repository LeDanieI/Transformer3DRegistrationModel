## Main
from datasets import OASISDataset
from plotimg import plotimg

import torch
import torch.utils.data as data


#This is a test

# torch.cuda.empty_cache()
# torch.backends.cudnn.benchmark = True     
learning_rate = 1e-9 # Tune this hyperparameter
batch_size = 1
epochs = 2
device = 'cpu'



dataset = OASISDataset()

train_set, val_set, test_set = data.random_split(dataset,[0.7,0.1,0.2], generator=torch.Generator().manual_seed(42))
