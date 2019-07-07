#
# Created on May 16, 2019
#
# @author: Julian Fortune
# @Description: Functions for unit testing and comparing changes.
#

import numpy as np

red = "\u001b[31m"
green = "\u001b[32m"
yellow = "\u001b[33m"
blue = "\u001b[34m"
reset = "\u001b[0m"

featureNames = ["time", "syllablesPerSecond", "meanVoiceActivity", "stDevVoiceActivity", "meanPitch", "stDevPitch", "meanIntensity", "stDevIntensity"]

# | Compares two numpy arrays and displays information about their differences.
def compareArrays(old, new):
    print()

    if np.array_equal(old, new):
        print(green + "  No changes!" + reset)
    else:
        numberOfColumnsInOld = old.shape[0]
        numberOfColumnsInNew = new.shape[0]

        for columnIndex in range(numberOfColumnsInOld):
            oldColumn = old[columnIndex]
            newColumn = new[columnIndex]

            if np.array_equal(oldColumn, newColumn):
                print(green + "  '" + featureNames[columnIndex] + "' unchanged" + reset)
            else:
                print(yellow + "~ '" + featureNames[columnIndex] + "' modified" + reset)

                numberChanged = newColumn.size - (oldColumn == newColumn).sum()
                differences = newColumn - oldColumn

                totalDifferent = np.subtract(newColumn, oldColumn)

                print(yellow + "    - " + str(numberChanged) + " out of "
                      + str(newColumn[1:].size) + " entries changed." + reset )
                print(yellow + "    - " +"Average change: " +
                str(differences.sum()/numberChanged) + ", Max: " +
                str(max(differences)) + reset)
    print()

# | Compares two numpy arrays saved in files.
def compareArrayFiles(oldFeaturesPath, newFeaturesPath):
    oldFeatures = np.load(oldFeaturesPath)
    newFeatures = np.load(newFeaturesPath)

    compareArrays(oldFeatures, newFeatures)
