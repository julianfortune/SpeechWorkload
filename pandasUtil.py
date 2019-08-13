#
# Created on August 12, 2018
#
# @author: Julian Fortune
# @Description: File for testing and converting to Pandas data storage.
#

import glob, sys, csv, os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def openNumpyFile():
    path = "./training/Supervisory_Evaluation_Day_2/features/current5seconds/"
    outPath = "./pandas/day2/"

    labels = []
    with open(path + "labels.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            labels = row

    for filePath in sorted(glob.iglob(path + "*.npy")):
        fileName = os.path.basename(filePath)[:-4]

        array = np.load(filePath)

        array = np.swapaxes(array, 0, 1)
        frame = pd.DataFrame(array, columns= labels)
        frame.to_csv(outPath + fileName + ".csv", index= False)

def convertModelFiles():
    path = "./training/Supervisory_Evaluation_Day_1/labels/"

    for filePath in sorted(glob.iglob(path + "*.npy")):
        fileName = os.path.basename(filePath)[:-4]

        array = np.load(filePath)
        frame = pd.DataFrame(array, columns= ["workload"])
        frame.index.name = "time"
        print(frame)

        frame.to_csv("./training/Supervisory_Evaluation_Day_1/labels/" + fileName + ".csv")

def convertTimeToSeconds(timeString):
    return int(timeString.split(":")[0]) * 60 + float(timeString.split(":")[1])

def createTimeIndexedLabelsFile():
    filePath = "./training/Supervisory_Evaluation_Day_1/labels/underload.csv"
    lengthOfEvaluationInSeconds = 900
    timeStep = 1

    fileFrame = pd.read_csv(filePath)
    print(fileFrame)

    fileIndex = 0

    workload = []
    times = np.arange(0, lengthOfEvaluationInSeconds, timeStep)

    for timeValue in times:
        while fileIndex + 1 < len(fileFrame) and convertTimeToSeconds(fileFrame.loc[fileIndex + 1]['Clock']) <= timeValue:
            fileIndex += 1
        workload.append(fileFrame.loc[fileIndex]['Speech'])

    workloadData = np.swapaxes(np.array([times, workload]), 0, 1)
    workloadFrame = pd.DataFrame(workloadData, columns= ['time', 'speechWorkload'])

    workloadFrame.to_csv("./training/Supervisory_Evaluation_Day_1/labels/ul.csv", index= False)

def main():
    createTimeIndexedLabelsFile()

main()
