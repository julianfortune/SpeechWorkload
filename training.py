#
# Created on Nov 12, 2018
#
# @author: Julian Fortune
# @Description: Functions for training and assessing the neural network.
#

import glob
import sys
import csv
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tflearn
import pandas as pd
from scipy import stats


def loadData(directory=None, trainingFiles=None, filter=True, threshold=0.1, audioFeatures=["meanIntensity", "stDevIntensity", "meanPitch", "stDevPitch", "stDevVoiceActivity", "meanVoiceActivity", "syllablesPerSecond", "filledPauses"], respirationRate=True, trimToRespirationLength=True, shouldGraph=False, labelsHaveCondition=False, labelsHaveOverallState=False):
    data = pd.DataFrame(columns=audioFeatures +
                                (["respirationRate"] if respirationRate else list()) +
                                ["speechWorkload"] +
                                (["condition"] if labelsHaveCondition else list()) +
                                (["overallState"] if labelsHaveOverallState else list()))

    files = []

    if directory and not trainingFiles:
        files = list(sorted(glob.iglob(directory + "features/*.csv")))
    elif trainingFiles and not directory:
        files = trainingFiles
    else:
        assert False, "Only directory or training files can be passed."

    for path in files:
        name = os.path.basename(path)[:-4]

        # print("Loading:", path)

        if name[0].lower() == 'p' and name[1:2].isdigit() and not("run_time" in name):
            currentData = pd.read_csv(path, index_col=0)
            currentLabels = pd.read_csv(path.replace(
                "features", "labels"), index_col=0)

            if len(currentLabels) != len(currentData.index):
                # print("WARNING: Shapes of labels and inputs do not match:",
                #       currentLabels.shape, currentData.shape, end=" ")

                # TODO: Keep or delete?
                # Trim data to be the length of the labels
                currentData = currentData.iloc[0:len(currentLabels.index), :]

                currentLabels = currentLabels.iloc[0:len(currentData.index), :]

                # print("New shapes: ",
                #       currentLabels.shape, currentData.shape)

            # Add the speech workload values

            # TODO: Add condition/workloadState to labels
            currentData['speechWorkload'] = currentLabels['speechWorkload']

            if labelsHaveCondition:
                assert 'condition' in currentLabels.columns, (name + "missing condition!")
                currentData['condition'] = currentLabels['condition']
            if labelsHaveOverallState:
                assert 'overall' in currentLabels.columns, (name + "missing overall workload state!")
                currentData['overallState'] = currentLabels['overall']

            # print(currentData)

            # Adjust the data to include respiration rate or be the length of the respiration rate data frame
            if respirationRate:
                physioPath = path.replace("features", "physiological")
                # Physiological data missing
                if not os.path.isfile(physioPath):
                    # print("Missing physio data for:", name, physioPath)
                    print(name)
                    continue

                respirationRateData = pd.read_csv(path.replace(
                    "features", "physiological"), index_col=0)
                # print("rrdata", respirationRateData)
                currentData["respirationRate"] = respirationRateData
                currentData = currentData.dropna()
            elif trimToRespirationLength:
                # If doing a comparison without respirationRate make sure the samples are the same
                respirationRateData = pd.read_csv(path.replace(
                    "features", "physiological"), index_col=0)
                currentData = currentData.iloc[0:len(
                    respirationRateData.index), :]

            if shouldGraph:
                currentData.plot()
                plt.title(name)
                plt.show()

            data = data.append(
                currentData[data.columns], ignore_index=True)

        else:
            print("Ignoring:", name)

    print("Using", list(data.columns))

    # The training data always needs this filtering
    if filter:
        data = data[(data['meanVoiceActivity'] > threshold)
                    & (data['speechWorkload'] > 0)]

    return data


def neuralNetwork(data, train=True, epochs=50):
    inputs = data.drop(columns=['speechWorkload']).to_numpy()
    labels = np.reshape(data['speechWorkload'].to_numpy(), (-1, 1))

    # Shuffle data
    inputs, labels = tflearn.data_utils.shuffle(inputs, labels)

    # Neural network characteristics
    input_neurons = inputs.shape[1]  # Size in the second dimension
    hidden_neurons = 256
    output_neurons = 1

    with tf.device('/gpu:0'):
        # Set up
        tf.reset_default_graph()
        tflearn.init_graph()

        # Input layer
        net = tflearn.input_data(shape=[None, input_neurons])

        # Hidden layers
        net = tflearn.fully_connected(
            net, hidden_neurons, bias=True, activation='relu')
        net = tflearn.fully_connected(
            net, hidden_neurons, bias=True, activation='relu')
        net = tflearn.fully_connected(
            net, hidden_neurons, bias=True, activation='relu')

        # Output layer
        net = tflearn.fully_connected(net, output_neurons)

        # Set the method for regression
        net = tflearn.regression(net, optimizer='Adam', learning_rate=0.001,
                                 loss='mean_square', metric='R2', restore=True, batch_size=64)

        # Create the model from the network
        model = tflearn.DNN(net, tensorboard_verbose=0)

        if train:
            # Fit the data, `validation_set=` sets asside a proportion of the data to validate with
            model.fit(inputs, labels, n_epoch=epochs,
                      validation_set=0.10, show_metric=True)

        return model

def accuracyMetrics(model, data):
    # Catch empty data set before bad things happen D:
    if not len(data) > 0:
        return [len(data.index), None, None, None, None, None, None, None, None, None, None, None, None, None]

    # Remove any extra information about overall condition/workload state to
    # not mess up this function
    if "condition" in data.columns:
        data = data.drop(columns=['condition'])
    if "overallState" in data.columns:
        data = data.drop(columns=['overallState'])

    predictions = model.predict(data.drop(columns=['speechWorkload']).to_numpy())[:, 0]
    predictions[data.meanVoiceActivity < 0.1] = 0

    actual = data.speechWorkload.to_numpy()

    if len(actual) >= 2:
        correlationCoefficient, significance = stats.pearsonr(actual, predictions)
    else:
        correlationCoefficient, significance = (None, None)
    rmse = np.sqrt(np.mean((predictions - actual)
                           ** 2, axis=0, keepdims=True))[0]
    actualMean = np.mean(actual)
    actualStandardDeviation = np.std(actual)
    actualMedian = np.median(actual)
    actualMinimum = actual.min()
    actualMaximum = actual.max()

    predictionsMean = np.mean(predictions)
    predictionsStandardDeviation = np.std(predictions)
    predictionsMedian = np.median(predictions)
    predictionsMinimum = predictions.min()
    predictionsMaximum = predictions.max()

    # print([len(data.index), correlationCoefficient, significance, rmse,
    #        actualMean, actualStandardDeviation, predictionsMean, predictionsStandardDeviation])

    return [len(data.index), correlationCoefficient, significance, rmse,
           actualMean, actualStandardDeviation, actualMedian, actualMinimum, actualMaximum,
           predictionsMean, predictionsStandardDeviation, predictionsMedian, predictionsMinimum, predictionsMaximum]


def assessModelAccuracy(model, data, shouldFilterOutMismatch=False, shouldGraph=False, shouldSplitByWorkloadState=False, shouldSplitByCondition=False):
    assessmentData = data

    if shouldFilterOutMismatch:
        vocalData = assessmentData[(assessmentData["speechWorkload"] > 0) & (
            assessmentData["meanVoiceActivity"] >= 0.1)]
        vocalData = vocalData.reset_index().drop(columns=["index"])

        silentData = assessmentData[(assessmentData["speechWorkload"] == 0) & (
            assessmentData["meanVoiceActivity"] < 0.1)]
        silentData = silentData.reset_index().drop(columns=["index"])

        if len(silentData) > len(vocalData):
            silentData = silentData.sample(
                n=len(vocalData.index), random_state=930123201)
            silentData = silentData.reset_index().drop(columns=["index"])
        else:
            vocalData = vocalData.sample(
                n=len(silentData.index), random_state=930123201)
            vocalData = vocalData.reset_index().drop(columns=["index"])

        assessmentData = pd.concat([vocalData, silentData])
        assessmentData = assessmentData.reset_index().drop(columns=["index"])

    if not len(assessmentData) > 0:
        if shouldSplitByWorkloadState: # Prevent unpacking from failing.
            return [[len(assessmentData.index), None, None, None, None, None, None, None, None, None, None, None, None, None]] * 3

        if shouldSplitByCondition: # Prevent unpacking from failing.
            return [[len(assessmentData.index), None, None, None, None, None, None, None, None, None, None, None, None, None]] * 2

        return [len(assessmentData.index), None, None, None, None, None, None, None, None, None, None, None, None, None]

    # print(assessmentData)
    if shouldSplitByWorkloadState:
        overload = assessmentData[(assessmentData["overallState"] == "ol")]
        normalLoad = assessmentData[(assessmentData["overallState"] == "nl")]
        underload = assessmentData[(assessmentData["overallState"] == "ul")]

        return accuracyMetrics(model, overload), accuracyMetrics(model, normalLoad), accuracyMetrics(model, underload)

    if shouldSplitByCondition:
        high = assessmentData[(assessmentData["condition"] == "high")]
        low = assessmentData[(assessmentData["condition"] == "low")]

        return accuracyMetrics(model, high), accuracyMetrics(model, low)


    return accuracyMetrics(model, assessmentData)



# Emulated Real-World Conditions - Train Day1, test Day2
def supervisoryRealWorld(epochs, leaveOut=[], trainModelsAndSave=True):
    features = ["meanIntensity", "stDevIntensity", "meanPitch", "stDevPitch", "stDevVoiceActivity",
                "meanVoiceActivity", "syllablesPerSecond", "filledPauses", "respirationRate"]

    for featureToLeaveOut in leaveOut:
        features.remove(featureToLeaveOut)

    includeRespirationRate = "respirationRate" in features

    audioFeatures = features
    if includeRespirationRate:
        audioFeatures.remove("respirationRate")

    modelDirectory = "./models/Supervisory_Real_World/"
    day1Directory = "./training/Supervisory_Evaluation_Day_1/"
    day2Directory = "./training/Supervisory_Evaluation_Day_2/"

    if len(leaveOut) > 0:
        leaveOutString = str(leaveOut).replace(
            "[", "").replace("]", "").replace("'", "").replace(", ", "-")
        modelDirectory = "./models/Supervisory_Real_World-LeaveOut-" + leaveOutString + "/"

    train = loadData(directory=day1Directory, audioFeatures=audioFeatures,
                     respirationRate=includeRespirationRate)
    test = loadData(directory=day2Directory, audioFeatures=audioFeatures,
                    respirationRate=includeRespirationRate, filter=False,
                    labelsHaveOverallState=True)

    if trainModelsAndSave:
        model = neuralNetwork(train, epochs=epochs)
        model.save(modelDirectory + "realWorld-" +
                   str(epochs) + "epochs.tflearn")
    else:
        model = neuralNetwork(train, train=False)
        model.load(modelDirectory + "realWorld-" +
                   str(epochs) + "epochs.tflearn")


    # Assess the performance of the neural network
    nonSplit = [[False, "all"] + assessModelAccuracy(model, test)]
    nonSplitFiltered = [[True, "all"] + assessModelAccuracy(model, test, shouldFilterOutMismatch=True)]

    # Asses by each condition (low/high)
    ol, nl, ul = assessModelAccuracy(model, test, shouldSplitByWorkloadState=True)
    unfilteredSplit = [[False, "ol"] + ol, [False, "nl"] + nl, [False, "ul"] + ul]

    olFilter, nlFilter, ulFilter = assessModelAccuracy(model, test, shouldFilterOutMismatch=True, shouldSplitByWorkloadState=True)
    filteredSplit = [[True, "ol"] + olFilter, [True, "nl"] + nlFilter, [True, "ul"] + ulFilter]

    metrics = unfilteredSplit + nonSplit + filteredSplit + nonSplitFiltered

    # Convert results to data frame
    results = pd.DataFrame(metrics, columns=["filtered", "overallWorkloadState", "samples", "coefficient",
                                             "significance", "RMSE", "actualMean", "actualStDev", "actualMedian", "actualMin", "actualMax", "predMean", "predStDev", "predMedian", "predMin", "predMax"])

    print(results)
    results.to_csv("./analyses/realWorldResults-LeaveOut" +
                   str(leaveOut) + "-" + str(epochs) + "epochs.csv")


# Population Generalizability
def supervisoryLeaveOneOutCrossValidation(epochs, leaveOut=[], trainModelsAndSave=True):
    features = ["meanIntensity", "stDevIntensity", "meanPitch", "stDevPitch", "stDevVoiceActivity",
                "meanVoiceActivity", "syllablesPerSecond", "filledPauses", "respirationRate"]

    for featureToLeaveOut in leaveOut:
        features.remove(featureToLeaveOut)

    includeRespirationRate = "respirationRate" in features

    audioFeatures = features
    if includeRespirationRate:
        audioFeatures.remove("respirationRate")

    modelDirectory = "./models/Supervisory_Leave_One_Out/"
    day1Directory = "./training/Supervisory_Evaluation_Day_1/"
    day2Directory = "./training/Supervisory_Evaluation_Day_2/"

    if len(leaveOut) > 0:
        leaveOutString = str(leaveOut).replace(
            "[", "").replace("]", "").replace("'", "").replace(", ", "-")
        modelDirectory = "./models/Supervisory_Leave_One_Out-LeaveOut-" + leaveOutString + "/"

    results = pd.DataFrame(columns=["participant", "filtered", "overallWorkloadState", "samples", "coefficient",
                                    "significance", "RMSE", "actualMean", "actualStDev", "actualMedian", "actualMin", "actualMax", "predMean", "predStDev", "predMedian", "predMin", "predMax"])

    for participantNumber in range(1, 31):
        print("Participant", participantNumber)

        # Files containing data from the current participant under investigation.
        # To be used for testing (leave-one-out cross-val).
        participantPaths = []

        # Grab all files from the evaluation.
        featurePaths = list(sorted(glob.iglob(day1Directory + "features/*.csv"))) + \
            list(sorted(glob.iglob(day2Directory + "features/*.csv")))

        # Identify paths to files for the current participant under investigation.
        for path in featurePaths:
            if str(participantNumber) == os.path.basename(path).split('_')[0][1:]:
                participantPaths.append(path)

        # Remove the participant under investigation's files from the training set.
        for path in participantPaths:
            featurePaths.remove(path)

        train = loadData(trainingFiles=featurePaths, audioFeatures=audioFeatures,
                         respirationRate=includeRespirationRate)
        test = loadData(trainingFiles=participantPaths, audioFeatures=audioFeatures,
                        respirationRate=includeRespirationRate, filter=False,
                        labelsHaveOverallState=True)

        if trainModelsAndSave:
            model = neuralNetwork(train, epochs=epochs)
            model.save(modelDirectory + "leaveOut-" +
                       str(participantNumber) + "-" + str(epochs) + "epochs.tflearn")
        else:
            model = neuralNetwork(train, train=False)
            model.load(modelDirectory + "leaveOut-" +
                       str(participantNumber) + "-" + str(epochs) + "epochs.tflearn")

        # Append results to the end of the data frame
        # TODO: Make results from each into a pd frame and use concat(ignore_index=True) to append to overall `results`
        # Assess the performance of the neural network
        nonSplit         = [[participantNumber, False, "all"] + assessModelAccuracy(model, test)]
        nonSplitFiltered = [[participantNumber, True, "all"] + assessModelAccuracy(model, test, shouldFilterOutMismatch=True)]

        # Asses by each condition (low/high)
        ol, nl, ul = assessModelAccuracy(model, test, shouldSplitByWorkloadState=True)
        unfilteredSplit = [[participantNumber, False, "ol"] + ol,
                        [participantNumber, False, "nl"] + nl,
                        [participantNumber, False, "ul"] + ul]

        olFilter, nlFilter, ulFilter = assessModelAccuracy(model, test, shouldFilterOutMismatch=True, shouldSplitByWorkloadState=True)
        filteredSplit                = [[participantNumber, True, "ol"] + olFilter,
                                        [participantNumber, True, "nl"] + nlFilter,
                                        [participantNumber, True, "ul"] + ulFilter]

        metrics = unfilteredSplit + nonSplit + filteredSplit + nonSplitFiltered

        # Convert results to data frame
        currentResults = pd.DataFrame(metrics, columns=results.columns)

        # print("CURRENT RESULTS --- ")
        print(currentResults)

        results = pd.concat([results, currentResults], ignore_index=True)

        # currentResults =
        # results.loc[len(results)] = [participantNumber, False, "all"] + assessModelAccuracy(model, test)
        # results.loc[len(results)] = [participantNumber, True, "all"] + assessModelAccuracy(model, test, shouldFilterOutMismatch=True)

    print(results)
    results.to_csv("./analyses/supervisoryCrossValidationResults-LeaveOut" +
                   str(leaveOut) + "-" + str(epochs) + "epochs.csv")
    summary = createSummary(results)
    summary.to_csv("./analyses/supervisoryCrossValidationResults-LeaveOut" +
                   str(leaveOut) + "-" + str(epochs) + "epochs-summary.csv")


# Human-Robot Teaming Generalizability - Train on Supervisory, test on Peer-Based (split by low/condition)
def supervisoryHumanRobot(epochs, leaveOut=[], trainModelsAndSave=True):
    features = ["meanIntensity", "stDevIntensity", "meanPitch", "stDevPitch", "stDevVoiceActivity",
                "meanVoiceActivity", "syllablesPerSecond", "filledPauses", "respirationRate"]

    for featureToLeaveOut in leaveOut:
        features.remove(featureToLeaveOut)

    includeRespirationRate = "respirationRate" in features

    audioFeatures = features
    if includeRespirationRate:
        audioFeatures.remove("respirationRate")

    modelDirectory = "./models/Supervisory_Human_Robot/"

    # Training
    day1Directory = "./training/Supervisory_Evaluation_Day_1/"
    day2Directory = "./training/Supervisory_Evaluation_Day_2/"

    # Testing
    peerDirectory = "./training/Peer_Based/"

    if len(leaveOut) > 0:
        leaveOutString = str(leaveOut).replace(
            "[", "").replace("]", "").replace("'", "").replace(", ", "-")
        modelDirectory = "./models/Supervisory_Human_Robot-LeaveOut-" + leaveOutString + "/"

    trainDay1 = loadData(directory=day1Directory, audioFeatures=audioFeatures,
                         respirationRate=includeRespirationRate, trimToRespirationLength=False)
    trainDay2 = loadData(directory=day2Directory, audioFeatures=audioFeatures,
                         respirationRate=includeRespirationRate, trimToRespirationLength=False)
    train = pd.concat([trainDay1, trainDay2])

    test = loadData(directory=peerDirectory, audioFeatures=audioFeatures,
                    trimToRespirationLength=False, respirationRate=includeRespirationRate,
                    filter=False, labelsHaveCondition=True)

    # print(test)

    if trainModelsAndSave:
        model = neuralNetwork(train, epochs=epochs)
        model.save(modelDirectory + "supervisoryHumanRobot" +
                   str(epochs) + "epochs.tflearn")
    else:
        model = neuralNetwork(train, train=False)
        model.load(modelDirectory + "supervisoryHumanRobot" +
                   str(epochs) + "epochs.tflearn")

    # Assess the performance of the neural network
    nonSplit = [[False, "both"] + assessModelAccuracy(model, test)]
    nonSplitFiltered = [[True, "both"] + assessModelAccuracy(model, test, shouldFilterOutMismatch=True)]

    # Asses by each condition (low/high)
    low, high = assessModelAccuracy(model, test, shouldSplitByCondition=True)
    unfilteredSplit = [[False, "low"] + low, [False, "high"] + high]

    lowFiltered, highFiltered = assessModelAccuracy(model, test, shouldFilterOutMismatch=True, shouldSplitByCondition=True)
    filteredSplit = [[True, "low"] + lowFiltered, [True, "high"] + highFiltered]

    metrics = unfilteredSplit + nonSplit + filteredSplit + nonSplitFiltered

    # Convert results to data frame
    results = pd.DataFrame(metrics, columns=["filtered", "condition", "samples", "coefficient",
                                             "significance", "RMSE", "actualMean", "actualStDev", "actualMedian", "actualMin", "actualMax", "predMean", "predStDev", "predMedian", "predMin", "predMax"])

    # print(results)
    results.to_csv("./analyses/supervisoryHumanRobot-LeaveOut" +
                   str(leaveOut) + "-" + str(epochs) + "epochs.csv")


# Human-Robot Teaming Generalizability - Train on Peer-Based, test on Supervisory
def peerHumanRobot(epochs, leaveOut=[], trainModelsAndSave=True):
    features = ["meanIntensity", "stDevIntensity", "meanPitch", "stDevPitch", "stDevVoiceActivity",
                "meanVoiceActivity", "syllablesPerSecond", "filledPauses", "respirationRate"]

    for featureToLeaveOut in leaveOut:
        features.remove(featureToLeaveOut)

    includeRespirationRate = "respirationRate" in features

    audioFeatures = features
    if includeRespirationRate:
        audioFeatures.remove("respirationRate")

    modelDirectory = "./models/Peer_Human_Robot/"

    # Training
    peerDirectory = "./training/Peer_Based/"

    # Testing
    day1Directory = "./training/Supervisory_Evaluation_Day_1/"
    day2Directory = "./training/Supervisory_Evaluation_Day_2/"

    if len(leaveOut) > 0:
        leaveOutString = str(leaveOut).replace(
            "[", "").replace("]", "").replace("'", "").replace(", ", "-")
        modelDirectory = "./models/Peer_Human_Robot-LeaveOut-" + leaveOutString + "/"
        # modelDirectory.replace("\'", "\\\'")

    # print(modelDirectory, os.path.exists(modelDirectory))

    train = loadData(directory=peerDirectory, audioFeatures=audioFeatures, trimToRespirationLength=False,
                     respirationRate=includeRespirationRate)

    testDay1 = loadData(directory=day1Directory, audioFeatures=audioFeatures,
                        respirationRate=includeRespirationRate, trimToRespirationLength=False, filter=False, labelsHaveOverallState=True)
    testDay2 = loadData(directory=day2Directory, audioFeatures=audioFeatures,
                        respirationRate=includeRespirationRate, trimToRespirationLength=False, filter=False, labelsHaveOverallState=True)
    test = pd.concat([testDay1, testDay2],  ignore_index=True)

    if trainModelsAndSave:
        model = neuralNetwork(train, epochs=epochs)
        model.save(modelDirectory + "peerHumanRobot" +
                   str(epochs) + "epochs.tflearn")
    else:
        model = neuralNetwork(train, train=False)
        model.load(modelDirectory + "peerHumanRobot" +
                   str(epochs) + "epochs.tflearn")

    # Assess the performance of the neural network
    nonSplit = [[False, "all"] + assessModelAccuracy(model, test)]
    nonSplitFiltered = [[True, "all"] + assessModelAccuracy(model, test, shouldFilterOutMismatch=True)]

    # Asses by each condition (low/high)
    ol, nl, ul = assessModelAccuracy(model, test, shouldSplitByWorkloadState=True)
    unfilteredSplit = [[False, "ol"] + ol, [False, "nl"] + nl, [False, "ul"] + ul]

    olFilter, nlFilter, ulFilter = assessModelAccuracy(model, test, shouldFilterOutMismatch=True, shouldSplitByWorkloadState=True)
    filteredSplit = [[True, "ol"] + olFilter, [True, "nl"] + nlFilter, [True, "ul"] + ulFilter]

    metrics = unfilteredSplit + nonSplit + filteredSplit + nonSplitFiltered

    # Convert results to data frame
    results = pd.DataFrame(metrics, columns=["filtered", "overallWorkloadState", "samples", "coefficient",
                                             "significance", "RMSE", "actualMean", "actualStDev", "actualMedian", "actualMin", "actualMax", "predMean", "predStDev", "predMedian", "predMin", "predMax"])

    # print(results)
    results.to_csv("./analyses/peerHumanRobot-LeaveOut" +
                   str(leaveOut) + "-" + str(epochs) + "epochs.csv")


# Real-time evaluation sanity check - Train on Supervisory, test on Real-Time
def realTimeSanityCheck(epochs, leaveOut=[], trainModelsAndSave=True):
    features = ["meanIntensity", "stDevIntensity", "meanPitch", "stDevPitch", "stDevVoiceActivity",
                "meanVoiceActivity", "syllablesPerSecond", "filledPauses", "respirationRate"]

    for featureToLeaveOut in leaveOut:
        features.remove(featureToLeaveOut)

    includeRespirationRate = "respirationRate" in features

    audioFeatures = features
    if includeRespirationRate:
        audioFeatures.remove("respirationRate")

    modelDirectory = "./models/Real_Time_Sanity/"

    # Training
    day1Directory = "./training/Supervisory_Evaluation_Day_1/"
    day2Directory = "./training/Supervisory_Evaluation_Day_2/"

    # Testing
    realTimeDirectory = "./training/Real_Time/"

    if len(leaveOut) > 0:
        leaveOutString = str(leaveOut).replace(
            "[", "").replace("]", "").replace("'", "").replace(", ", "-")
        modelDirectory = "./models/Real_Time_Sanity-LeaveOut-" + leaveOutString + "/"

    trainDay1 = loadData(directory=day1Directory, audioFeatures=audioFeatures,
                         respirationRate=includeRespirationRate, trimToRespirationLength=False)
    trainDay2 = loadData(directory=day2Directory, audioFeatures=audioFeatures,
                         respirationRate=includeRespirationRate, trimToRespirationLength=False)
    train = pd.concat([trainDay1, trainDay2])

    test = loadData(directory=realTimeDirectory, audioFeatures=audioFeatures,
                trimToRespirationLength=False, respirationRate=includeRespirationRate,
                filter=False, labelsHaveOverallState=True)

    if trainModelsAndSave:
        model = neuralNetwork(train, epochs=epochs)
        model.save(modelDirectory + "realTimeSanityCheck" +
                   str(epochs) + "epochs.tflearn")
    else:
        model = neuralNetwork(train, train=False)
        model.load(modelDirectory + "realTimeSanityCheck" +
                   str(epochs) + "epochs.tflearn")

    metrics = [[False] + assessModelAccuracy(model, test), [
        True] + assessModelAccuracy(model, test, shouldFilterOutMismatch=True)]

    # Assess the performance of the neural network
    nonSplit = [[False, "all"] + assessModelAccuracy(model, test)]
    nonSplitFiltered = [[True, "all"] + assessModelAccuracy(model, test, shouldFilterOutMismatch=True)]

    # Asses by each condition (low/high)
    ol, nl, ul = assessModelAccuracy(model, test, shouldSplitByWorkloadState=True)
    unfilteredSplit = [[False, "ol"] + ol, [False, "nl"] + nl, [False, "ul"] + ul]

    olFilter, nlFilter, ulFilter = assessModelAccuracy(model, test, shouldFilterOutMismatch=True, shouldSplitByWorkloadState=True)
    filteredSplit = [[True, "ol"] + olFilter, [True, "nl"] + nlFilter, [True, "ul"] + ulFilter]

    metrics = unfilteredSplit + nonSplit + filteredSplit + nonSplitFiltered

    # Convert results to data frame
    results = pd.DataFrame(metrics, columns=["filtered", "overallWorkloadState", "samples", "coefficient",
                                             "significance", "RMSE", "actualMean", "actualStDev", "actualMedian", "actualMin", "actualMax", "predMean", "predStDev", "predMedian", "predMin", "predMax"])

    # print(results)
    results.to_csv("./analyses/realTimeSanityCheck-LeaveOut" +
                   str(leaveOut) + "-" + str(epochs) + "epochs.csv")


# Real-time window size evaluation
def realTimeWindowSizeEvaluation(epochs, leaveOut=[], trainModelsAndSave=True):
    features = ["meanIntensity", "stDevIntensity", "meanPitch", "stDevPitch", "stDevVoiceActivity",
                "meanVoiceActivity", "syllablesPerSecond", "filledPauses", "respirationRate"]

    for featureToLeaveOut in leaveOut:
        features.remove(featureToLeaveOut)

    includeRespirationRate = "respirationRate" in features

    audioFeatures = features
    if includeRespirationRate:
        audioFeatures.remove("respirationRate")

    directories = {
        1: "Real_Time-1_second_window",
        5: "Real_Time-5_second_window",
        10: "Real_Time-10_second_window",
        15: "Real_Time-15_second_window",
        30: "Real_Time-30_second_window",
        60: "Real_Time-60_second_window"}

    results = pd.DataFrame(columns=["windowSize", "participant", "filtered", "overallWorkloadState", "samples", "coefficient",
                                    "significance", "RMSE", "actualMean", "actualStDev", "actualMedian", "actualMin", "actualMax", "predMean", "predStDev", "predMedian", "predMin", "predMax"])

    for windowSize, directory in directories.items():
        featureDirectory = "./training/" + directory + "/"

        modelDirectory = "./models/Real_Time_Window_Size-" + \
            str(windowSize) + "_seconds/"

        if len(leaveOut) > 0:
            leaveOutString = str(leaveOut).replace(
                "[", "").replace("]", "").replace("'", "").replace(", ", "-")
            modelDirectory = "./models/Real_Time_Window_Size-" + \
            str(windowSize) + "_seconds-LeaveOut-" + leaveOutString + "/"

        print(modelDirectory)

        for participantNumber in range(1, 31):
            print("Participant", participantNumber, "windowSize:", windowSize)

            participantPath = ""

            filePathsInDirectory = list(
                sorted(glob.iglob(featureDirectory + "features/*.csv")))

            featurePaths = [path for path in filePathsInDirectory if (os.path.basename(path)[:-4][0] == 'p'
                                                                      and os.path.basename(path)[:-4][1:].isdigit())]

            for path in featurePaths:
                if str(participantNumber) == os.path.basename(path).split('_')[0][1:][:-4]:
                    participantPath = path

            if participantPath != "":  # some participants are missing from the dataset
                featurePaths.remove(participantPath)

                train = loadData(trainingFiles=featurePaths, audioFeatures=audioFeatures,
                                 respirationRate=includeRespirationRate, trimToRespirationLength=False)
                test = loadData(trainingFiles=[participantPath], audioFeatures=audioFeatures,
                                respirationRate=includeRespirationRate, trimToRespirationLength=False, filter=False,
                                labelsHaveOverallState=True)

            if trainModelsAndSave:
                model = neuralNetwork(train, epochs=epochs)
                model.save(modelDirectory + "leaveOut-" +
                           str(participantNumber) + "-" + str(epochs) + "epochs.tflearn")
            else:
                model = neuralNetwork(train, train=False)
                model.load(modelDirectory + "leaveOut-" +
                           str(participantNumber) + "-" + str(epochs) + "epochs.tflearn")


            # Assess the performance of the neural network
            nonSplit         = [[windowSize, participantNumber, False, "all"] + assessModelAccuracy(model, test)]
            nonSplitFiltered = [[windowSize, participantNumber, True, "all"] + assessModelAccuracy(model, test, shouldFilterOutMismatch=True)]

            # Asses by each condition (low/high)
            ol, nl, ul      = assessModelAccuracy(model, test, shouldSplitByWorkloadState=True)
            unfilteredSplit = [[windowSize, participantNumber, False, "ol"] + ol,
                               [windowSize, participantNumber, False, "nl"] + nl,
                               [windowSize, participantNumber, False, "ul"] + ul]

            olFilter, nlFilter, ulFilter = assessModelAccuracy(model, test, shouldFilterOutMismatch=True, shouldSplitByWorkloadState=True)
            filteredSplit                = [[windowSize, participantNumber, True, "ol"] + olFilter,
                                            [windowSize, participantNumber, True, "nl"] + nlFilter,
                                            [windowSize, participantNumber, True, "ul"] + ulFilter]

            metrics = unfilteredSplit + nonSplit + filteredSplit + nonSplitFiltered

            # Convert results to data frame
            currentResults = pd.DataFrame(metrics, columns=results.columns)

            print(currentResults)

            results = pd.concat([results, currentResults], ignore_index=True)

            # print(">>> about to append ")
            # # Append results to the end of the data frame
            # print(results)
            # # TODO: Make results from each into a pd frame and use concat(ignore_index=True) to append to overall `results`
            # results.loc[len(results)] = [windowSize, participantNumber,
            #                              False] + assessModelAccuracy(model, test)
            # results.loc[len(results)] = [windowSize, participantNumber, True] + \
            #     assessModelAccuracy(model, test, shouldFilterOutMismatch=True)

        print(results)

    results.to_csv("./analyses/realTimeWindowSizeEvaluation-LeaveOut" +
                   str(leaveOut) + "-" + str(epochs) + "epochs.csv")

    summary = createSummary(results)
    summary.to_csv("./analyses/realTimeWindowSizeEvaluation-LeaveOut" +
                   str(leaveOut) + "-" + str(epochs) + "epochs-summary.csv")


#                          pd.DataFrame   Bool  String
def summarizeWorkloadState(dataFrame, filtered, state):
    workloadStateData = dataFrame.loc[(dataFrame["filtered"] == filtered) & (dataFrame["overallWorkloadState"] == state)]

    meanDataFrame = workloadStateData.mean()
    # Use total number of samples instead of mean
    meanDataFrame["samples"] = workloadStateData.sum()["samples"]
    # Set values not defined for mean()
    meanDataFrame["participant"] = "all"
    meanDataFrame["overallWorkloadState"] = state
    meanDataFrame["filtered"] = filtered

    return meanDataFrame

def createSummary(dataFrame):
    dataFrame = dataFrame.dropna()

    if "windowSize" in dataFrame.columns:
        resultFrame = pd.DataFrame([], columns=dataFrame.columns)

        for windowSize in [1, 5, 10, 15, 30, 60]:
            windowSizeData = dataFrame.loc[dataFrame["windowSize"]
                                           == windowSize]

            filteredOverload = summarizeWorkloadState(windowSizeData, True, "ol")
            filteredNormalLoad = summarizeWorkloadState(windowSizeData, True, "nl")
            filteredUnderload = summarizeWorkloadState(windowSizeData, True, "ul")
            filtered = summarizeWorkloadState(windowSizeData, True, "all")

            unfilteredOverload = summarizeWorkloadState(windowSizeData, False, "ol")
            unfilteredNormalLoad = summarizeWorkloadState(windowSizeData, False, "nl")
            unfilteredUnderload = summarizeWorkloadState(windowSizeData, False, "ul")
            unfiltered = summarizeWorkloadState(windowSizeData, False, "all")

            resultFrame = pd.concat([resultFrame,
                                     pd.DataFrame([filteredOverload,
                                                   filteredNormalLoad,
                                                   filteredUnderload,
                                                   filtered,
                                                   unfilteredOverload,
                                                   unfilteredNormalLoad,
                                                   unfilteredUnderload,
                                                   unfiltered],
                                                  columns=dataFrame.columns)])

        return resultFrame

    else:
        filteredOverload = summarizeWorkloadState(dataFrame, True, "ol")
        filteredNormalLoad = summarizeWorkloadState(dataFrame, True, "nl")
        filteredUnderload = summarizeWorkloadState(dataFrame, True, "ul")
        filtered = summarizeWorkloadState(dataFrame, True, "all")

        unfilteredOverload = summarizeWorkloadState(dataFrame, False, "ol")
        unfilteredNormalLoad = summarizeWorkloadState(dataFrame, False, "nl")
        unfilteredUnderload = summarizeWorkloadState(dataFrame, False, "ul")
        unfiltered = summarizeWorkloadState(dataFrame, False, "all")

        resultFrame = pd.DataFrame([filteredOverload,
                                    filteredNormalLoad,
                                    filteredUnderload,
                                    filtered,
                                    unfilteredOverload,
                                    unfilteredNormalLoad,
                                    unfilteredUnderload,
                                    unfiltered],
                                    columns=dataFrame.columns)

        return resultFrame


def main():
    print(); print(); print()

    # supervisoryRealWorld(100, trainModelsAndSave=False)
    # supervisoryRealWorld(100, trainModelsAndSave=False,
    #                      leaveOut=["filledPauses"])
    # supervisoryRealWorld(100, trainModelsAndSave=False,
    #                      leaveOut=["respirationRate"])
    # supervisoryRealWorld(100, trainModelsAndSave=False,
    #                      leaveOut=["respirationRate", "filledPauses"])

    # supervisoryLeaveOneOutCrossValidation(50, trainModelsAndSave=False)
    # supervisoryLeaveOneOutCrossValidation(50, trainModelsAndSave=False, leaveOut=["filledPauses"])
    # supervisoryLeaveOneOutCrossValidation(50, trainModelsAndSave=False, leaveOut=["respirationRate"])
    supervisoryLeaveOneOutCrossValidation(50, trainModelsAndSave=True, leaveOut=["respirationRate", "filledPauses"])

    # supervisoryHumanRobot(100, trainModelsAndSave=False)
    # supervisoryHumanRobot(100, trainModelsAndSave=False, leaveOut=["filledPauses"])
    # supervisoryHumanRobot(100, trainModelsAndSave=False, leaveOut=["respirationRate"])
    # supervisoryHumanRobot(100, trainModelsAndSave=False, leaveOut=["respirationRate", "filledPauses"])

    # peerHumanRobot(100, trainModelsAndSave=False)
    # peerHumanRobot(100, trainModelsAndSave=False, leaveOut=["filledPauses"])
    # peerHumanRobot(100, trainModelsAndSave=False, leaveOut=["respirationRate"])
    # peerHumanRobot(100, trainModelsAndSave=False, leaveOut=["respirationRate", "filledPauses"])

    # realTimeSanityCheck(100, trainModelsAndSave=False)
    # realTimeSanityCheck(100, trainModelsAndSave=False,
    #                     leaveOut=["filledPauses"])
    # realTimeSanityCheck(100, trainModelsAndSave=False,
    #                     leaveOut=["respirationRate"])
    # realTimeSanityCheck(100, trainModelsAndSave=False,
    #                     leaveOut=["respirationRate", "filledPauses"])

    # # TODO
    # realTimeWindowSizeEvaluation(50, trainModelsAndSave= False)
    # realTimeWindowSizeEvaluation(50, trainModelsAndSave= False, leaveOut= ["respirationRate"])
    # realTimeWindowSizeEvaluation(50, trainModelsAndSave= False, leaveOut= ["filledPauses"])
    # realTimeWindowSizeEvaluation(50, trainModelsAndSave= False, leaveOut= ["filledPauses", "respirationRate"])


if __name__ == "__main__":
    main()
