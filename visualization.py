import numpy, os, glob
import matplotlib.pyplot as plt # Visualisation
import pandas as pd

def plotFeatures():
    dir = "./training/Supervisory_Evaluation_Day_1/features/*.csv"

    for path in sorted(glob.iglob(dir)):
        name = os.path.basename(path)[:-4]
        features = pd.read_csv(path)

        features.drop(columns=['time']).plot(subplots= True, figsize= (14, 8), title= name)
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,
                            wspace=0.5)
        plt.savefig("./figures/" + name + ".png")
        plt.close()

def main():
    plotFeatures()

main()
