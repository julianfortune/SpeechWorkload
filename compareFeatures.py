from speechLibrary import testModule
import numpy, os, glob
import matplotlib.pyplot as plt # Visualisation

import csv

def graphParticipants():
    testDir = "./training/Supervisory_Evaluation_Day_1/features/current5second/"
    oldDir =  "./training/Supervisory_Evaluation_Day_1/features/HFES/"

    oldLabels = []
    with open(oldDir + 'labels.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            oldLabels = row

    testLabels = []
    with open(testDir + 'labels.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            testLabels = row

    for testPath in sorted(glob.iglob(testDir + "*.npy")):
        oldPath = oldDir + os.path.basename(testPath)

        newFeatures = numpy.load(testPath)
        oldFeatures = numpy.load(oldPath)

        name = os.path.basename(oldPath)[:-4]

        testModule.graphArrays(oldLabels= oldLabels,
                               oldFeatures= oldFeatures,
                               newLabels= testLabels,
                               newFeatures= newFeatures,
                               filePath= "./figures/" + name,
                               name= name)

def compareParticipant():
    participant = 'p10_nl'

    oldDir = "./features/HFES/"
    testDir =  "./features/current30second/"

    oldLabels = []
    with open(oldDir + 'labels.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            oldLabels = row

    testLabels = []
    with open(testDir + 'labels.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            testLabels = row

    testFeatures = numpy.load(testDir + participant + '.npy')
    oldFeatures = numpy.load(oldDir  + participant + '.npy')

    testModule.compareArrays(oldLabels= oldLabels,
                             oldFeatures= oldFeatures,
                             newLabels= testLabels,
                             newFeatures= testFeatures)


def main():
    graphParticipants()

main()
