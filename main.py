## Main
from datasets import OASISDataset
from plotimg import plotimg
from model.configurations import get_VitBase_config
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
config = get_VitBase_config(img_size=tuple(dataset.inshape))
model = RegTransformer(config)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch