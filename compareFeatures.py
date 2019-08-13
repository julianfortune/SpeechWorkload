#
# Created on May 16, 2019
#
# @author: Julian Fortune
# @Description: Functions for unit testing and comparing changes.
#

import os, glob, csv

import matplotlib.pyplot as plt # Visualisation
import numpy as np
import pandas as pd

RED = "\u001b[31m"
GREEN = "\u001b[32m"
YELLOW = "\u001b[33m"
BLUE = "\u001b[34m"
RESET = "\u001b[0m"

# | Compares two numpy arrays and displays information about their differences.
def compareArrays(oldData, newData):
    # Make sure no funny business with feature arrays
    assert len(oldData.index) == len(newData.index), "Feature array lengths differ."

    print()

    if oldData.equals(newData):
        print(GREEN + "  No changes!" + RESET)

    else:
        # Check for features modified or removed from old array
        for featureName in list(oldData.columns):
            if featureName in list(newData.columns):
                oldColumn = oldData[featureName].to_numpy()
                newColumn = newData[featureName].to_numpy()

                # print(oldColumn, newColumn)

                if np.array_equal(oldColumn, newColumn):
                    print(GREEN + "  '" + featureName + "' unchanged" + RESET)

                else:
                    print(YELLOW + "~ '" + featureName + "' modified" + RESET)

                    numberChanged = newColumn.size - (oldColumn == newColumn).sum()
                    differences = abs(newColumn - oldColumn)

                    print(YELLOW + "    - " + str(numberChanged) + " out of "
                          + str(newColumn[1:].size) + " entries changed." + RESET )
                    print(YELLOW + "    - " + "Average change: " +
                          str(differences.sum()/numberChanged) + ", Max: " +
                          str(max(differences)) + RESET)
            else:
                print(RED + "- '" + featureName + "' removed" + RESET)

        # Check for features added to new array
        for featureName in list(newData.columns):
            if featureName not in list(oldData.columns):
                print(BLUE + "+ '" + featureName + "' added" + RESET)

    print()

# | Compares two numpy arrays and displays graphs.
def graphArrays(oldData, newData, filePath, name=None):
    # Make sure no funny business with feature arrays
    assert oldData.shape[1] == newData.shape[1], "Feature array lengths differ."
    assert "time" in oldData.columns, "Time array missing from first position."

    oldData = oldData.drop(columns=["time"])
    newData = newData.drop(columns=["time"])

    ax = oldData.plot(subplots= True, figsize=(16, 10), color= 'steelblue')
    newData.plot(ax= ax, subplots= True, color= 'darkorange')
    plt.subplots_adjust(hspace = 1)
    plt.title(name)
    plt.savefig(filePath + ".png")
    plt.close()

def graphParticipants():
    oldDir = "./training/Supervisory_Evaluation_Day_1/current30second/"
    testDir =  "./training/Supervisory_Evaluation_Day_1/features/"

    for testPath in sorted(glob.iglob(testDir + "*.csv")):
        name = os.path.basename(testPath)[:-4]
        oldPath = oldDir + os.path.basename(testPath)

        testData = pd.read_csv(testPath)
        oldData = pd.read_csv(oldPath)

        graphArrays(oldData= oldData,
                    newData= testData,
                    filePath= "./figures/comparison/" + name,
                    name= name)

def compareParticipant():
    participant = 'p10_nl'

    oldDir = "./training/Supervisory_Evaluation_Day_1/HFES/"
    testDir =  "./training/Supervisory_Evaluation_Day_1/current30second/"

    testData = pd.read_csv(testDir + participant + ".csv")
    oldData = pd.read_csv(oldDir  + participant + ".csv")

    compareArrays(oldData= oldData,
                  newData= testData)

def main():
    compareParticipant()

main()
