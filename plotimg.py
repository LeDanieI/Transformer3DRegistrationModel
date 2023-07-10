import matplotlib.pyplot as plt


def plotimg(dataset, index):
    fig, ax = plt.subplots()
    figure(figsize=(4, 3), dpi=300)
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2.50
    p1,=plt.plot(mseu9, color='black')
    p2,=plt.plot(mseu8, color='blue')
