import json
import scienceplots
import matplotlib as mpl
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np

def plotOutput(outputdic):
    plt.style.use(['science','ieee'])
    plt.style.use(['science','no-latex'])
    
   
    
    
    
    # plt.ylim(-15, 40)
    # plt.xlim(0, 180)
    for keys in outputdic.keys():
        fig, ax = plt.subplots()
        
        figure(figsize=(4, 3), dpi=300)
        plt.rcParams["axes.edgecolor"] = "black"
        plt.rcParams["axes.linewidth"] = 2.50
        p1,=plt.plot(outputdic[keys], color='black', label=keys)

        
        # p4,=plt.plot(mse9,linestyle='--', color='black')
        # p5,=plt.plot(mse8,linestyle='--', color='blue')
        # p6,=plt.plot(mse7,linestyle='--', color='red')
        
        plt.ylabel('Score',fontweight='bold')
        plt.xlabel('Image number (-)',fontweight='bold')
        plt.title(keys)
        # plt.legend(['MSE unsupervised','MSE supervised','LNCC'])
        # plt.legend([(p1, p4), (p2, p5)], ['1e-9 unsup/sup', '1e-8 unsup/sup'], numpoints=1,handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=3)
        plt.show()
