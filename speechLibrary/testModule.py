#
# Created on May 16, 2019
#
# @author: Julian Fortune
# @Description: Functions for unit testing and comparing changes.
#

import numpy as np

RED = "\u001b[31m"
GREEN = "\u001b[32m"
YELLOW = "\u001b[33m"
BLUE = "\u001b[34m"
RESET = "\u001b[0m"

# | Compares two numpy arrays and displays information about their differences.
def compareArrays(oldLables, oldFeatures, newLabels, newFeatures):
    # Make sure labels match up in size to features
    assert len(oldLables) == oldFeatures.shape[0], "Old features do not match labels."
    assert len(newLabels) == newFeatures.shape[0], "New features do not match labels."

    # Make sure no funny business with feature arrays
    assert oldFeatures.shape[1] == newFeatures.shape[1], "Feature array lengths differ."

    print()

    if np.array_equal(oldFeatures, newFeatures):
        print(GREEN + "  No changes!" + RESET)

    else:
        # Check for features modified or removed from old array
        for featureName in oldLables:
            if featureName in newLabels:
                oldColumn = oldFeatures[oldLables.index(featureName)]
                newColumn = newFeatures[newLabels.index(featureName)]

                if np.array_equal(oldColumn, newColumn):
                    print(GREEN + "  '" + featureName + "' unchanged" + RESET)

                else:
                    print(YELLOW + "~ '" + featureName + "' modified" + RESET)

                    numberChanged = newColumn.size - (oldColumn == newColumn).sum()
                    differences = newColumn - oldColumn

                    print(YELLOW + "    - " + str(numberChanged) + " out of "
                          + str(newColumn[1:].size) + " entries changed." + RESET )
                    print(YELLOW + "    - " + "Average change: " +
                          str(differences.sum()/numberChanged) + ", Max: " +
                          str(max(differences)) + RESET)
            else:
                print(RED + "- '" + featureName + "' removed" + RESET)

        # Check for features added to new array
        for featureName in newLabels:
            if featureName not in oldLables:
                print(BLUE + "+ '" + featureName + "' added" + RESET)

    print()
