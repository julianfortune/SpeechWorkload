#
# Created on August 12, 2018
#
# @author: Julian Fortune
# @Description: File for creating Pandas .csv files from workload and phsysio data.
#

import glob, sys, csv, os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

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
        return float(timeString.split(":")[-2]) * 60 + float(timeString.split(":")[-1])

def createDay1LabelFiles():
    modelsPath = "../media/Jamison_Evaluations/Supervisory/Day1/Models/"
    lengthOfEvaluationInSeconds = 900
    timeStep = 1

    featuresPath = "./training/Supervisory_Evaluation_Day_1/features/"

    for filePath in sorted(glob.iglob(featuresPath + "*.csv")):
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

    featuresPath = "./training/Supervisory_Evaluation_Day_2/features/"

    for filePath in sorted(glob.iglob(featuresPath + "*.csv")):
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

def createPeerBasedLabelFiles():
    modelsPath = "../media/Jamison_Evaluations/Models/"
    featuresPath = "./training/Peer_Based/features/"

    timeStep = 1

    # Model dataframes
    T1HighData = pd.read_csv(modelsPath + "t1_high.csv")
    T1LowData =  pd.read_csv(modelsPath + "t1_low.csv")

    T2HighData = pd.read_csv(modelsPath + "t2_high.csv")
    T2LowData =  pd.read_csv(modelsPath + "t2_low.csv")

    T3HighData = pd.read_csv(modelsPath + "t3_high.csv")
    T3LowData =  pd.read_csv(modelsPath + "t3_low.csv")

    T4HighData = pd.read_csv(modelsPath + "t4_high.csv")
    T4LowData =  pd.read_csv(modelsPath + "t4_low.csv")

    # Data on which participant gets which model file
    T1HighParticipants = ['P3', 'P4', 'P5', 'P8', 'P12', 'P14', 'P16', 'P18']
    T1LowParticipants = ['P2', 'P6', 'P7', 'P9', 'P10', 'P15', 'P17', 'P13']

    T2HighParticipants = ['P2', 'P4', 'P5', 'P8', 'P9', 'P10', 'P13', 'P14']
    T2LowParticipants = ['P3', 'P6', 'P7', 'P12', 'P15', 'P16', 'P17', 'P18']

    T3HighParticipants = ['P4', 'P5', 'P12', 'P6', 'P7', 'P9', 'P10']
    T3LowParticipants = ['P2', 'P3', 'P8', 'P13', 'P14', 'P15', 'P16', 'P17']

    T4HighParticipants = ['P2', 'P4',  'P8', 'P9', 'P12', 'P15', 'P16', 'P7']
    T4LowParticipants = ['P3', 'P5', 'P6', 'P13', 'P14', 'P17', 'P18', 'P10']

    for filePath in sorted(glob.iglob(featuresPath + "*.csv")):
        fileName = os.path.basename(filePath)[:-4]
        participantName = os.path.basename(filePath)[:-4].split("_")[0]
        task = os.path.basename(filePath)[:-4].split("_")[1]

        modelData = None

        if task == "T1":
            if participantName in T1HighParticipants:
                modelData = T1HighData
            elif participantName in T1LowParticipants:
                modelData = T1LowData
            else:
                print("ERROR: Unaccounted for participant", participantName)
        if task == "T2":
            if participantName in T2HighParticipants:
                modelData = T2HighData
            elif participantName in T2LowParticipants:
                modelData = T2LowData
            else:
                print("ERROR: Unaccounted for participant", participantName)
        if task == "T3":
            if participantName in T3HighParticipants:
                modelData = T3HighData
            elif participantName in T3LowParticipants:
                modelData = T3LowData
            else:
                print("ERROR: Unaccounted for participant", participantName)
        if task == "T4":
            if participantName in T4HighParticipants:
                modelData = T4HighData
            elif participantName in T4LowParticipants:
                modelData = T4LowData
            else:
                print("ERROR: Unaccounted for participant", participantName)

        fileIndex = 0
        workload = []

        lengthOfEvaluationInSeconds = int(convertTimeToSeconds(list(modelData["Clock"])[len(modelData.index) - 1][3:]))

        times = np.arange(0, lengthOfEvaluationInSeconds, timeStep)

        for timeValue in times:
            while fileIndex + 1 < len(modelData) and round(convertTimeToSeconds(modelData.loc[fileIndex + 1]['Clock'])) <= timeValue:
                fileIndex += 1
            workload.append(modelData.loc[fileIndex]['Speech'])
            # print(timeValue, fileIndex, modelData.loc[fileIndex]['Speech'])

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

        workloadFrame.to_csv("./training/Peer_Based/labels/" + fileName + ".csv", index= False)

def createRealTimeLabelFiles():
    modelsPath = "../media/Jamison_Evaluations/Real_Time_Evaluation/"
    featuresPath = "./training/Real_Time/features/"

    timeStep = 1

    for filePath in sorted(glob.iglob(featuresPath + "*.csv")):
        fileName = os.path.basename(filePath)[:-4]

        print(fileName)

        participantNumber = os.path.basename(filePath)[:-4].split("_")[0][1:]

        if participantNumber == "19":
            modelPath = modelsPath + "Participant_" + participantNumber + "/p" + participantNumber + "_baseline_wl.csv"
        else:
            modelPath = modelsPath + "Participant_" + participantNumber + "/p" + participantNumber + "_wl.csv"

        print(modelPath)

        modelData = pd.read_csv(modelPath)
        modelData.rename(columns={"Unnamed: 0": "Clock"}, inplace=True )

        fileIndex = 0
        workload = []

        clock = pd.to_datetime(modelData["Clock"]).astype(np.int64).to_numpy() / 1000000000
        clock = [element - clock[0] for element in clock]

        modelData["Clock"] = clock

        lengthOfEvaluationInSeconds = math.ceil(list(modelData["Clock"])[len(modelData.index) - 1])

        times = np.arange(0, lengthOfEvaluationInSeconds, timeStep)

        for timeValue in times:
            while fileIndex + 1 < len(modelData) and round(modelData.loc[fileIndex + 1]['Clock']) <= timeValue:
                fileIndex += 1
            workload.append(modelData.loc[fileIndex]['Speech'])
            # print(timeValue, fileIndex, modelData.loc[fileIndex]['Speech'])

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

        workloadFrame.to_csv("./training/Real_Time/labels/" + fileName + ".csv", index= False)

def makeSupervisoryPhysioDataTrainingFiles():
    physio = pd.read_csv("../media/Jamison_Evaluations/Supervisory/physiological.csv")[[ "Participant", "Condition", "Seconds", "Respiration Rate"]]
    print(physio)

    day1featuresPath = "./training/Supervisory_Evaluation_Day_1/features/"
    day2featuresPath = "./training/Supervisory_Evaluation_Day_2/features/"

    features = list(sorted(glob.iglob(day2featuresPath + "*.csv"))) + list(sorted(glob.iglob(day1featuresPath + "*.csv")))

    for filePath in features:
        fileName = os.path.basename(filePath)[:-4]
        participant = "p" + os.path.basename(filePath)[:-4].split("_")[0][1:]
        condition = os.path.basename(filePath)[:-4].split("_")[1]
        print(participant, condition)

        data = physio[(physio["Participant"] == participant) & (physio["Condition"] == condition)][["Seconds", "Respiration Rate"]]
        data = data.reset_index().drop(columns= ["index"]).set_index("Seconds")
        data.index.name = "time"
        data = data.rename(columns={"Respiration Rate": "respirationRate"})

        if condition == "day2":
            data.to_csv("./training/Supervisory_Evaluation_Day_2/physiological/" + fileName + ".csv")
        else:
            data.to_csv("./training/Supervisory_Evaluation_Day_1/physiological/" + fileName + ".csv")

def makeRealTimePhysioDataTrainingFiles():
    directory = "../media/Jamison_Evaluations/Real_Time_Evaluation/"
    outputDirectoryPath = "/"

    for participantNumber in range(1, 32):
        if participantNumber not in [16, 19]:
            physioPath = directory + "Participant_" + str(participantNumber) + "/p" + str(participantNumber) + "_physio.csv"
            data = pd.read_csv(physioPath)

            data = data.reset_index().drop(columns= ["index"])
            data.index.name = "time"
            data = data.rename(columns={"Respiration_Rate": "respirationRate"})
            data = data[["respirationRate"]]

            # Fill in missing NaN values with linear interpolation
            data = data.interpolate(limit_direction='both')

            data.to_csv("./training/Real_Time/physiological/" + "p" + str(participantNumber) + ".csv")

def makePeerBasedPhysioDataTrainingFiles():
    # directory = "../media/Jamison_Evaluations/Peer_Based/"
    physio = pd.read_csv("../media/Jamison_Evaluations/Peer_Based/peer_physio.csv")[[ "Participant", "Seconds", "Condition", "Respiration Rate"]]
    print(physio)


    for participantNumber in range(1, 32):
        # if participantNumber not in [16, 19]:
        for conditionNumber in [1,2,3,4]:
            name = 'P' + str(participantNumber)
            condition = 'T' + str(conditionNumber)
            # print(name)

            data = physio[(physio["Participant"] == name) & (physio["Condition"] == condition)][["Seconds", "Respiration Rate"]]
            #

            if not data.empty:
                print(name, '_', condition)

                data = data.reset_index().drop(columns= ["index"]).set_index("Seconds")
                data.index.name = "time"
                data = data.rename(columns={"Breathing Data": "respirationRate"})

                print(data)

                fileName = str(name) + '_' + str(condition)

                data.to_csv("./training/Peer_Based/physiological/" + fileName + ".csv")

    #     if condition == "day2":
    #
    #     else:
    #         data.to_csv("./training/Supervisory_Evaluation_Day_1/physiological/" + fileName + ".csv")

def main():
    makeSupervisoryPhysioDataTrainingFiles()

main()
