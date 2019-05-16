import numpy as np

red = "\u001b[31m"
green = "\u001b[32m"
yellow = "\u001b[33m"
blue = "\u001b[34m"
reset = "\u001b[0m"

featureNames = ["time", "voiceActivity", "filledPauses"]

def compareArrays(old, new):
    print()

    if np.array_equal(old, new):
        print(green + "  No changes!" + reset)
    else:
        numberOfColumnsInOld = oldFeatures.shape[1]
        numberOfColumnsInNew = newFeatures.shape[1]

        for columnIndex in range(numberOfColumnsInOld):
            oldColumn = old[:,columnIndex]
            newColumn = new[:,columnIndex]

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

def compareArrayFiles(oldFeaturesFile, newFeatures):
    oldFeatures = numpy.load(oldFeaturesPath)
    newFeatures = numpy.load(newFeaturesPath)

    compareArrays(oldFeatures, newFeatures)
