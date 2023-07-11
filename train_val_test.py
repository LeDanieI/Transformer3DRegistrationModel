"""
Training/validation functions
"""
import sys
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from utils import NCC, DSC
import monai

from tkinter import filedialog as fd
from tkinter import Tk
import pdb

def train_epoch(model, data_loader, dataset, optimizer, device):
    """
    Train for one epoch
    """
    total_ncc_batch = 0
    total_mse_batch = 0
    total_dsc_batch = 0
    total_mse_img_batch = 0
    total_hd95_batch = 0
    
    # Initialize loss functions
    similarity_loss = NCC(device)
    mse_loss = torch.nn.MSELoss()
    dsc = DSC(device)
    
    model.train()
    for batch_idx, (img_moving, img_fixed, mask_moving, mask_fixed, T_ground_truth, T_augment) in enumerate(tqdm(data_loader, file=sys.stdout)):
        # Take the img_moving and fixed images to the GPU
        img_moving, img_fixed, mask_moving, mask_fixed = img_moving.to(device).unsqueeze(0), img_fixed.to(device).unsqueeze(0), mask_moving.to(device).unsqueeze(0), mask_fixed.to(device).unsqueeze(0)
        optimizer.zero_grad(set_to_none=True)################

        img_warped, T = model(img_moving, img_fixed)
        mask_warped = dataset.transform_rigid(T,mask_moving)
        
        loss = similarity_loss.forward(img_fixed, img_warped)
        T_error = mse_loss(T, T_ground_truth.to(device))
        MSE_img = mse_loss(img_warped, img_fixed)
        dice = dsc.forward(mask_warped, mask_fixed)
        # hd95 = monai.metrics.compute_hausdorff_distance(mask_warped.unsqueeze(0), mask_fixed, percentile=95)
        # print(loss)
        total_ncc_batch += loss.item()
        total_mse_batch += T_error.item()
        total_dsc_batch += dice
        total_mse_img_batch += MSE_img.item()
        # total_hd95_batch += hd95
        
        T_error.backward() ######## CHANGE LOSS FUNCTION TYPE HERE
        optimizer.step()
        del loss, T_error, img_moving, img_fixed, img_warped, T_ground_truth, dice, mask_moving, mask_fixed, mask_warped, MSE_img


    train_ncc_loss = total_ncc_batch / len(data_loader)   
    train_T_error = total_mse_batch / len(data_loader)
    train_dsc = total_dsc_batch / len(data_loader)
    mse_img = total_mse_img_batch / len(data_loader)
    # hd95_train = total_hd95_batch / len(data_loader)
    
    """ Print loss """
    print("Train Loss = %.5f" % train_T_error)
    return train_ncc_loss, train_T_error, train_dsc, mse_img

def validate_epoch(model, val_loader, dataset, device):

    
    val_ncc_batch = 0
    val_T_error_batch = 0
    total_dsc_batch = 0
    total_mse_img_batch = 0
    total_hd95_batch = 0

    mse_loss = torch.nn.MSELoss()
    dsc = DSC(device)
    similarity_loss = NCC(device)
    
    #Disable training
    model.train(mode=False)
    torch.no_grad()
    
    for batch_idx, (img_moving, img_fixed, mask_moving, mask_fixed, T_ground_truth, T_augment) in enumerate(tqdm(val_loader, file=sys.stdout)):
        img_moving, img_fixed, mask_moving, mask_fixed = img_moving.to(device).unsqueeze(0), img_fixed.to(device).unsqueeze(0), mask_moving.to(device).unsqueeze(0), mask_fixed.to(device).unsqueeze(0)

        img_warped, T = model(img_moving, img_fixed)
        mask_warped = dataset.transform_rigid(T,mask_moving)

        val_loss = similarity_loss.forward(img_fixed, img_warped)  
        dice = dsc.forward(mask_warped, mask_fixed)
        MSE_img = mse_loss(img_warped, img_fixed).item()
        T_error = mse_loss(T, T_ground_truth.to(device))
        # hd95 = monai.metrics.compute_hausdorff_distance(mask_warped.unsqueeze(0), mask_fixed, percentile=95)

        val_ncc_batch += val_loss.item()
        val_T_error_batch += T_error.item()
        total_dsc_batch += dice
        total_mse_img_batch += MSE_img
        # total_hd95_batch += hd95

        del val_loss, img_moving, img_fixed, img_warped, T_error, T_ground_truth, dice, mask_moving, mask_fixed, mask_warped, MSE_img

    val_ncc_loss = val_ncc_batch/len(val_loader)
    val_T_error = val_T_error_batch /len(val_loader)
    val_dsc = total_dsc_batch / len(val_loader)
    mse_img = total_mse_img_batch / len(val_loader)
    # hd95_val = total_hd95_batch / len(val_loader)

    print("Validation Loss = %.5f" % val_T_error)
    return val_ncc_loss, val_T_error, val_dsc, mse_img


def test_model(model, test_loader, dataset, device):
    """ TESTING """
    root = Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    # Path = fd.askopenfilename()  
    weightspath = fd.askopenfilename(parent=root)
    # torch.save(model.state_dict(), f'save/supervised/{learning_rate}/weights.pth')
    model.load_state_dict(torch.load(weightspath))
    
    out={}
    list_out = ['ncc','MSE_T','MSE_img','dice','hd95']
    for key in list_out:
        out[key]=[]
        
    mse_loss = torch.nn.MSELoss()
    dsc = DSC(device)
    similarity_loss = NCC(device)
    
    #Disable training
    model.train(mode=False)
    torch.no_grad()
    with torch.no_grad():
        for batch_idx, (img_moving, img_fixed, mask_moving, mask_fixed, T_ground_truth, T_augment) in enumerate(tqdm(test_loader, file=sys.stdout)):
            img_moving, img_fixed, mask_moving, mask_fixed = img_moving.to(device).unsqueeze(0), img_fixed.to(device).unsqueeze(0), mask_moving.to(device).unsqueeze(0), mask_fixed.to(device).unsqueeze(0)
            img_warped, T = model(img_moving, img_fixed)
            mask_warped = dataset.transform_rigid(T,mask_moving)
    
            test_loss = similarity_loss.forward(img_fixed, img_warped)  
            dice = dsc.forward(mask_warped, mask_fixed)
            MSE_img = mse_loss(img_warped, img_fixed).item()
            T_error = mse_loss(T, T_ground_truth.to(device)).item()
            hd95 = monai.metrics.compute_hausdorff_distance(mask_warped.unsqueeze(0), mask_fixed, percentile=95)
    
            scores_img = [test_loss, T_error, MSE_img, dice, hd95]
            for key, score in zip(list_out,scores_img):
                out[key].append(score)
    
            del test_loss, img_moving, img_fixed, img_warped, T_error, T_ground_truth, dice, mask_moving, mask_fixed, mask_warped, MSE_img, hd95, key, score

    return out

def test_initial(model, test_loader, dataset, device):
    initial_ncc = []
    initial_dsc = []
    initial_mse_img = []
    #initial_mse_T = []
    initial_hd95 = []
    
    mse_loss = torch.nn.MSELoss()
    dsc = DSC(device)
    similarity_loss = NCC(device)
    
    #Disable training
    model.train(mode=False)
    torch.no_grad()
    
    for batch_idx, (img_moving, img_fixed, mask_moving, mask_fixed, T_ground_truth, T_augment) in enumerate(tqdm(test_loader, file=sys.stdout)):
        img_moving, img_fixed, mask_moving, mask_fixed = img_moving.to(device), img_fixed.to(device), mask_moving.to(device), mask_fixed.to(device)
 
        img_warped, T = model(img_moving, img_fixed)
        
        initial_ncc += [similarity_loss.forward(img_fixed, img_moving)]
        initial_dsc += [dsc.forward(mask_moving, mask_fixed)]
        initial_mse_img += [mse_loss(img_moving, img_fixed).item()]
        #initial_mse_T += [mse_loss(T.squeeze(), T_augment.to(device).squeeze()).item()]
        initial_hd95 += [monai.metrics.compute_hausdorff_distance(mask_moving, mask_fixed, percentile=95)]
        
        

        del img_moving, img_fixed, mask_moving, mask_fixed
    initialavg = [initial_ncc, initial_dsc, initial_mse_img, initial_hd95]
    # test_ncc_loss = test_ncc_batch/len(test_loader)
    # test_T_error = test_T_error_batch /len(test_loader)
    # test_dsc = total_dsc_batch / len(test_loader)
    # mse_img = total_mse_img_batch / len(test_loader)
    # hd95_test = total_hd95_batch / len(test_loader)
    return initialavg

def plot_test(model, test_set, dataset, plotlist, slicenr, device, modelname):
    similarity_loss = NCC(device)
    # test_loss_batch = 0 
    mse_loss = torch.nn.MSELoss()
    dsc=DSC(device)
    model.train(mode=False)
    torch.no_grad()

    
    for idx,i in enumerate(tqdm(plotlist,file=sys.stdout)):
        fig, axs = plt.subplots(1, 3)
        img_moving, img_fixed = test_set[i][0].to(device), test_set[i][1].to(device)
        img_moving, img_fixed = img_moving.unsqueeze(0), img_fixed.unsqueeze(0)
        img_warped, T = model(img_moving, img_fixed)
        
        mse = mse_loss(img_fixed, img_warped) 
        initial_mse = mse_loss(img_fixed,img_moving)
        
        ncc = similarity_loss.forward(img_fixed, img_warped)
        initial_ncc = similarity_loss.forward(img_fixed, img_moving)
        mask_moving = test_set[i][2]
        mask_fixed = test_set[i][3]
        mask_warped = dataset.transform_rigid(T,mask_moving.unsqueeze(0).to(device))

        dice = dsc.forward(mask_warped, mask_fixed)
        dice_initial = dsc.forward(mask_moving, mask_fixed)
        # print(mask_warped.shape, mask_moving.shape, mask_fixed.shape)
        hd95 = monai.metrics.compute_hausdorff_distance(mask_warped.unsqueeze(0), mask_fixed.unsqueeze(0), percentile=95)
        hd95_initial = monai.metrics.compute_hausdorff_distance(mask_moving.unsqueeze(0), mask_fixed.unsqueeze(0), percentile=95)
        img_moving , img_fixed = img_moving.detach(), img_fixed.detach()
        axs[0].imshow(img_moving.squeeze().cpu().numpy()[:, slicenr, :], cmap='gray')
        axs[0].set_title('Moving image')
        axs[1].imshow(img_fixed.squeeze().cpu().numpy()[:,slicenr, :], cmap='gray')
        axs[1].set_title('Fixed image')
        axs[2].imshow(img_warped.squeeze().detach().cpu().numpy()[:,slicenr , :], cmap='gray')
        axs[2].set_title('Warped image')
        img_warped = img_warped.detach()
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        
        fig.suptitle(f'MSE: {round(initial_mse.item(),4)} | {round(mse.item(),4)} \nNCC: {round(initial_ncc.item(),4)} | {round(ncc.item(),4)} \nDSC: {round(dice_initial,4)} | {round(dice,4)} \nHD95: {round(hd95_initial.item(),4)} | {round(hd95.item(),4)}\n\n')
        plt.savefig(f'save/mse_unsupervised/{modelname}/oasis_{i}.png')
        plt.close()
        del img_moving, img_fixed, img_warped, T, mse, ncc, dice, mask_moving, mask_fixed, mask_warped
    fig.show()

    