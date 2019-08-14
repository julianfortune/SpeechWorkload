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

def convertTimeToSeconds(timeString):
    if len(timeString) == 5:
        return int(timeString.split(":")[0]) * 60 + float(timeString.split(":")[1])
    if len(timeString) == 8:
        return int(timeString.split(":")[1]) * 60 + float(timeString.split(":")[2])

def createDay1LabelFiles():
    modelsPath = "../media/Jamison_Evaluations/Supervisory/Day1/Models/"
    lengthOfEvaluationInSeconds = 900
    timeStep = 1

    feauresPath = "./training/Supervisory_Evaluation_Day_1/features/"

    for filePath in sorted(glob.iglob(feauresPath + "*.csv")):
        fileName = os.path.basename(filePath)[:-4]
        print(fileName)

        modelName = ""

        if "nl" in fileName:
            modelName = "normal_load.csv"
        elif "ul" in fileName:
            modelName = "underload.csv"
        elif "ol" in fileName:
            modelName = "overload.csv"

        fileFrame = pd.read_csv(modelsPath + modelName)

        fileIndex = 0

        workload = []
        times = np.arange(0, lengthOfEvaluationInSeconds, timeStep)

        for timeValue in times:
            while fileIndex + 1 < len(fileFrame) and round(convertTimeToSeconds(fileFrame.loc[fileIndex + 1]['Clock'])) <= timeValue:
                fileIndex += 1
            workload.append(fileFrame.loc[fileIndex]['Speech'])

        currentData = pd.read_csv(filePath, index_col= 0)

        # Add extra zeros to the labels if inputs run over the length
        if len(currentData.index) > len(workload):
            offsetSize = len(currentData.index) - len(workload)
            fill = np.zeros(offsetSize)
            workload = np.append(workload, fill)

        times = np.arange(0, len(workload), timeStep)

        workloadData = np.swapaxes(np.array([times, workload]), 0, 1)
        workloadFrame = pd.DataFrame(workloadData, columns= ['time', 'speechWorkload'])
        print(workloadFrame)

        workloadFrame.to_csv("./training/Supervisory_Evaluation_Day_1/labels/" + fileName + ".csv", index= False)

def createDay2LabelFiles():
    modelsPath = "../media/Jamison_Evaluations/Supervisory/Day2/Models/"
    timeStep = 1

    feauresPath = "./training/Supervisory_Evaluation_Day_2/features/"

    for filePath in sorted(glob.iglob(feauresPath + "*.csv")):
        fileName = os.path.basename(filePath)[:-4]
        participantNumber = int(os.path.basename(filePath)[:-4].split("_")[0][1:])

        modelName = ""

        if participantNumber <= 10:
            modelName = "day2_order1_fixed.csv"
        elif participantNumber <= 20:
            modelName = "day2_order2_fixed.csv"
        else:
            modelName = "day2_order3_fixed.csv"

        print(fileName, participantNumber, modelName)

        fileFrame = pd.read_csv(modelsPath + modelName)
        fileIndex = 0

        workload = []

        lengthOfEvaluationInSeconds = int(convertTimeToSeconds(list(fileFrame["Clock"])[len(fileFrame.index) - 1][3:]))

        times = np.arange(0, lengthOfEvaluationInSeconds, timeStep)

        for timeValue in times:
            while fileIndex + 1 < len(fileFrame) and round(convertTimeToSeconds(fileFrame.loc[fileIndex + 1]['Clock'])) <= timeValue:
                fileIndex += 1
            workload.append(fileFrame.loc[fileIndex]['Speech'])
            # print(timeValue, fileIndex, fileFrame.loc[fileIndex]['Speech'])

        currentData = pd.read_csv(filePath, index_col= 0)

        # Add extra zeros to the labels if inputs run over the length
        if len(currentData.index) > len(workload):
            offsetSize = len(currentData.index) - len(workload)
            fill = np.zeros(offsetSize)
            workload = np.append(workload, fill)

        workload = workload[:len(currentData.index)]

        times = np.arange(0, len(workload), timeStep)

        workloadData = np.swapaxes(np.array([times, workload]), 0, 1)
        workloadFrame = pd.DataFrame(workloadData, columns= ['time', 'speechWorkload'])
        print(workloadFrame)

        workloadFrame.to_csv("./training/Supervisory_Evaluation_Day_2/labels/" + fileName + ".csv", index= False)

def testPhysioData():
    physio = pd.read_csv("./training/sup_physio.csv")[[ "Participant", "Condition", "Seconds", "Breathing Data"]]
    print(physio)

    feauresPath = "./training/Supervisory_Evaluation_Day_2/features/"

    for filePath in sorted(glob.iglob(feauresPath + "*.csv")):
        fileName = os.path.basename(filePath)[:-4]
        participant = "p" + os.path.basename(filePath)[:-4].split("_")[0][1:]
        condition = os.path.basename(filePath)[:-4].split("_")[1]

        data = physio[(physio["Participant"] == participant) & (physio["Condition"] == condition)][["Seconds", "Breathing Data"]]
        data = data.reset_index().drop(columns= ["index"]).set_index("Seconds")
        print(data)

def main():
    testPhysioData()

main()
