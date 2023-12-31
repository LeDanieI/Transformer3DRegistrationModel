## Main test
from datasets import OASISDataset
from plotimg import plotimg
from model.configurations import get_VitBase_config
from model.RegistrationNetworks import RegTransformer
import torch
import torch.utils.data as data
import time
from train_val_test import train_epoch, validate_epoch, test_model, plot_test, test_initial
from plotOutput import plotOutput
from plotSample import plotSample
#This is a test

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True     
learning_rate = 1e-9 # Tune this hyperparameter
batch_size = 1
epochs = 2
device = 'cuda'


#%% data prep
dataset = OASISDataset()
train_set, val_set, test_set = data.random_split(dataset,[0.7,0.1,0.2], generator=torch.Generator().manual_seed(42))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,num_workers=0)


config = get_VitBase_config(img_size=tuple(dataset.inshape))
model = RegTransformer(config)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%%
""" TRAINING """
print('\n----- Training -----')
epoch = 1
start = time.time()
while epoch <= epochs:
    print(f'\n[epoch {epoch} / {epochs}]')
    train_ncc, train_MSEt , train_dice, mse_train = train_epoch(model, train_loader, dataset, optimizer, device)
    # train_NCC_list.append(train_ncc)
    # train_MSEt_list.append(train_MSEt)
    # train_dsc_list.append(train_dice)
    # train_mse_list.append(mse_train)
    #train_hd95_list.append(hd95_train)
    
    val_ncc, val_MSEt, val_dice, mse_val = validate_epoch(model, val_loader, dataset, device)
    # val_NCC_list.append(val_ncc)
    # val_MSEt_list.append(val_MSEt)
    # val_dsc_list.append(val_dice)
    # val_mse_list.append(mse_val)
    # #val_hd95_list.append(hd95_val)
    
    epoch += 1
end = time.time()
traintime = round(end - start)
print('Total time training: ', traintime, ' seconds')


#%% Output is a dictionary with keys: ['ncc','MSE_T','MSE_img','dice','hd95']
""" TESTING """
output = test_model(model, test_loader, dataset, device)
#%% Copy from GPU to CPU
for key, values in output.items():
    for i, value in enumerate(values):
        if isinstance(value, torch.Tensor):
            values[i] = value.item()

#%% Plot graphs of scores
plotOutput(output)    

#%% Plot sample
plotSample(model, dataset, test_set, 5, 40, device)