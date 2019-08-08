#
# Created on May 16, 2019
#
# @author: Julian Fortune
# @Description: Functions for unit testing and comparing changes.
#

import numpy as np
import matplotlib.pyplot as plt # Visualisation

RED = "\u001b[31m"
GREEN = "\u001b[32m"
YELLOW = "\u001b[33m"
BLUE = "\u001b[34m"
RESET = "\u001b[0m"

# | Compares two numpy arrays and displays information about their differences.
def compareArrays(oldLabels, oldFeatures, newLabels, newFeatures):
    # Make sure labels match up in size to features
    assert len(oldLabels) == oldFeatures.shape[0], "Old features do not match labels."
    assert len(newLabels) == newFeatures.shape[0], "New features do not match labels."

    # Make sure no funny business with feature arrays
    assert oldFeatures.shape[1] == newFeatures.shape[1], "Feature array lengths differ."

    print()

    if np.array_equal(oldFeatures, newFeatures):
        print(GREEN + "  No changes!" + RESET)

    else:
        # Check for features modified or removed from old array
        for featureName in oldLabels:
            if featureName in newLabels:
                oldColumn = oldFeatures[oldLabels.index(featureName)]
                newColumn = newFeatures[newLabels.index(featureName)]

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
        for featureName in newLabels:
            if featureName not in oldLabels:
                print(BLUE + "+ '" + featureName + "' added" + RESET)

    print()

# | Compares two numpy arrays and displays graphs.
def graphArrays(oldLabels, oldFeatures, newLabels, newFeatures, filePath, name=None):
    # Make sure labels match up in size to features
    assert len(oldLabels) == oldFeatures.shape[0], "Old features do not match labels."
    assert len(newLabels) == newFeatures.shape[0], "New features do not match labels."

    # Make sure no funny business with feature arrays
    assert oldFeatures.shape[1] == newFeatures.shape[1], "Feature array lengths differ."
    assert oldLabels.index("time") == 0, "Time array missing from first position."

    seconds = oldFeatures[0]

    numberOfPlots = len(oldLabels) - 1

    plt.figure(figsize=[16, 10])
    plt.suptitle(name)

    for featureIndex in range(1, len(oldLabels)):
        featureName = oldLabels[featureIndex]

        plt.subplot(numberOfPlots * 100 + 10 + featureIndex)
        plt.plot(seconds, oldFeatures[featureIndex])
        plt.title(featureName)

        if featureName in newLabels:
            newFeatureIndex = newLabels.index(featureName)

            plt.plot(seconds, newFeatures[newFeatureIndex])
            plt.legend(["old", "new"])

    plt.subplots_adjust(hspace = 1)
    plt.savefig(filePath + ".png")
    plt.close()
