import matplotlib.pyplot as plt


def plotimg(dataset, index, T=None, slicenr=40):
    if T==None:
        fig, axs = plt.subplots(1,2)
    else:      
        fig, axs = plt.subplots(1,3)
        #axs[2].imshow([:,slicenr , :], cmap='gray')
        #axs[2].set_title('Warped image')
        
    axs[0].imshow(dataset[index][0][:, slicenr, :], cmap='gray')
    axs[0].set_title('Moving image')
    axs[1].imshow(dataset[index][1][:,slicenr, :], cmap='gray')
    axs[1].set_title('Fixed image')
    for ax in axs:
      ax.set_xticks([])
      ax.set_yticks([])
      
    
# figure(figsize=(4, 3), dpi=300)
# plt.rcParams["axes.edgecolor"] = "black"
# plt.rcParams["axes.linewidth"] = 2.50
# p1,=plt.plot(mseu9, color='black')
# p2,=plt.plot(mseu8, color='blue')
# #plt.plot(mseu7, color='red')

