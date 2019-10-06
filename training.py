#
# Created on Nov 12, 2018
#
# @author: Julian Fortune
# @Description: Functions for training and assessing the neural network.
#

import glob, sys, csv, os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tflearn
import pandas as pd
from scipy import stats



def loadData(directory= None, trainingFiles= None, filter= True, threshold= 0.1, audioFeatures= ["meanIntensity", "stDevIntensity", "meanPitch", "stDevPitch", "stDevVoiceActivity", "meanVoiceActivity", "syllablesPerSecond", "filledPauses"], respirationRate= True, trimToRespirationLength= True, shouldGraph= False):
    data = pd.DataFrame(columns= audioFeatures + (["respirationRate"] if respirationRate else list()) + ["speechWorkload"])

    files = []

    if directory and not trainingFiles:
        files = list(sorted(glob.iglob(directory + "features/*.csv")))
    elif trainingFiles and not directory:
        files = trainingFiles
    else:
        assert False, "Only directory or training files can be passed."

    for path in files:
        name = os.path.basename(path)[:-4]

        if name[0] == 'p' and name[1:].isdigit():
            currentData = pd.read_csv(path, index_col= 0)
            currentLabels = pd.read_csv(path.replace("features", "labels"), index_col= 0)


            if len(currentLabels) == len(currentData.index):
                # Add the speech workload values
                currentData['speechWorkload'] = currentLabels

                # Adjust the data to include respiration rate or be the length of the respiration rate data frame
                if respirationRate:
                    respirationRateData = pd.read_csv(path.replace("features", "physiological"), index_col= 0)
                    print("rrdata", respirationRateData)
                    currentData["respirationRate"] = respirationRateData
                    currentData = currentData.dropna()
                elif trimToRespirationLength:
                    # If doing a comparison without respirationRate make sure the samples are the same
                    respirationRateData = pd.read_csv(path.replace("features", "physiological"), index_col= 0)
                    currentData = currentData.iloc[0:len(respirationRateData.index), :]

                if shouldGraph:
                    currentData.plot()
                    plt.title(name)
                    plt.show()

                data = data.append(currentData[data.columns], ignore_index= True)
            else:
                print("WARNING: Shapes of labels and inputs do not match.", currentLabels.shape, currentData.shape)

    # print("Using", list(data.columns))

    if filter:
        data = data[(data['meanVoiceActivity'] > threshold) & (data['speechWorkload'] > 0)]

    return data

def neuralNetwork(data, train= True, epochs= 50):
    inputs = data.drop(columns=['speechWorkload']).to_numpy()
    labels = np.reshape(data['speechWorkload'].to_numpy(), (-1, 1))

    # Shuffle data
    inputs, labels = tflearn.data_utils.shuffle(inputs, labels)

    # Neural network characteristics
    input_neurons = inputs.shape[1] # Size in the second dimension
    hidden_neurons = 256
    output_neurons = 1

    with tf.device('/gpu:0'):
        # Set up
        tf.reset_default_graph()
        tflearn.init_graph()

        # Input layer
        net = tflearn.input_data(shape=[None, input_neurons])

        # Hidden layers
        net = tflearn.fully_connected(net, hidden_neurons, bias=True, activation='relu')
        net = tflearn.fully_connected(net, hidden_neurons, bias=True, activation='relu')
        net = tflearn.fully_connected(net, hidden_neurons, bias=True, activation='relu')

        # Output layer
        net = tflearn.fully_connected(net, output_neurons)

        # Set the method for regression
        net = tflearn.regression(net, optimizer='Adam', learning_rate=0.001,  loss='mean_square', metric = 'R2', restore=True, batch_size=64)

        # Create the model from the network
        model = tflearn.DNN(net, tensorboard_verbose=0)

        if train:
            # Fit the data, `validation_set=` sets asside a proportion of the data to validate with
            model.fit(inputs, labels, n_epoch= epochs, validation_set=0.10, show_metric=True)

        return model

def assessModelAccuracy(model, data, shouldFilterOutMismatch= False, shouldGraph= False):
    assessmentData = data

    if shouldFilterOutMismatch:
        vocalData = assessmentData[(assessmentData["speechWorkload"] > 0) & (assessmentData["meanVoiceActivity"] >= 0.1)]
        vocalData = vocalData.reset_index().drop(columns= ["index"])

        silentData = assessmentData[(assessmentData["speechWorkload"] == 0) & (assessmentData["meanVoiceActivity"] < 0.1)]
        silentData = silentData.reset_index().drop(columns= ["index"])

        if len(silentData) > len(vocalData):
            silentData = silentData.sample(n= len(vocalData.index), random_state= 930123201)
            silentData = silentData.reset_index().drop(columns= ["index"])
        else:
            vocalData = vocalData.sample(n= len(silentData.index), random_state= 930123201)
            vocalData = vocalData.reset_index().drop(columns= ["index"])

        print(vocalData)
        print(silentData)

        assessmentData = pd.concat([vocalData, silentData])
        assessmentData = assessmentData.reset_index().drop(columns= ["index"])

    if not len(assessmentData) > 0:
        return [len(assessmentData.index), None, None, None, None, None, None]

    predictions = model.predict(assessmentData.drop(columns=['speechWorkload']).to_numpy())[:,0]
    predictions[assessmentData.meanVoiceActivity < 0.1] = 0

    actual = assessmentData.speechWorkload.to_numpy()

    results = pd.DataFrame()
    results['predictions'] = predictions
    results['actual'] = assessmentData.speechWorkload

    if shouldGraph:
        pd.concat([results, assessmentData.meanIntensity, assessmentData.meanVoiceActivity], axis=1).plot()
        plt.show()

    correlationCoefficient = stats.pearsonr(actual, predictions)[0]
    rmse = np.sqrt(np.mean((predictions - actual)**2, axis=0, keepdims=True))[0]
    actualMean = np.mean(actual)
    actualStandardDeviation = np.std(actual)
    predictionsMean = np.mean(predictions)
    predictionsStandardDeviation = np.std(predictions)

    return [len(assessmentData.index), correlationCoefficient, rmse, actualMean, actualStandardDeviation, predictionsMean, predictionsStandardDeviation]

# Emulated Real-World Conditions
def supervisoryRealWorld(epochs, leaveOut= [], trainModelsAndSave= True, respirationRate= True):
    features= ["meanIntensity", "stDevIntensity", "meanPitch", "stDevPitch", "stDevVoiceActivity", "meanVoiceActivity", "syllablesPerSecond", "filledPauses", "respirationRate"]

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
        modelDirectory = "./models/Supervisory_Real_World-LeaveOut" + str(leaveOut) + "/"

    train = loadData(directory= day1Directory, audioFeatures= audioFeatures, respirationRate= includeRespirationRate)
    test = loadData(directory= day2Directory, audioFeatures= audioFeatures, respirationRate= includeRespirationRate, filter=False)

    if trainModelsAndSave:
        model = neuralNetwork(train, epochs= epochs)
        model.save(modelDirectory + "realWorld-" + str(epochs) + "epochs.tflearn")
    else:
        model = neuralNetwork(train, train= False)
        model.load(modelDirectory + "realWorld-" + str(epochs) + "epochs.tflearn")

    metrics = [[False] + assessModelAccuracy(model, test), [True] + assessModelAccuracy(model, test, shouldFilterOutMismatch= True)]

    # Append results to the end of the data frame
    results = pd.DataFrame(metrics, columns=["filtered", "samples", "coefficient", "RMSE", "actualMean", "actualStDev", "predMean", "predStDev"])

    print(results)
    results.to_csv("./analyses/realWorldResults-LeaveOut" + str(leaveOut) + "-" + str(epochs) + "epochs.csv")

# Population Generalizability
def supervisoryLeaveOneOutCrossValidation(epochs, leaveOut= [], trainModelsAndSave= True, respirationRate= True):
    features = ["meanIntensity", "stDevIntensity", "meanPitch", "stDevPitch", "stDevVoiceActivity", "meanVoiceActivity", "syllablesPerSecond", "filledPauses", "respirationRate"]

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
        modelDirectory = "./models/Supervisory_Leave_One_Out-LeaveOut" + str(leaveOut) + "/"

    results = pd.DataFrame(columns=["participant", "filtered", "samples", "coefficient", "RMSE", "actualMean", "actualStDev", "predMean", "predStDev"])

    participants = []

    for participantNumber in range(1, 31):
        print("Participant", participantNumber)

        participantPaths = []

        featurePaths = list(sorted(glob.iglob(day1Directory + "features/*.csv"))) + list(sorted(glob.iglob(day2Directory + "features/*.csv")))

        for path in featurePaths:
            if str(participantNumber) == os.path.basename(path).split('_')[0][1:]:
                participantPaths.append(path)

        for path in participantPaths:
            featurePaths.remove(path)

        train = loadData(trainingFiles=featurePaths, audioFeatures= audioFeatures, respirationRate= includeRespirationRate)
        test = loadData(trainingFiles=participantPaths, audioFeatures= audioFeatures, respirationRate= includeRespirationRate, filter=False)

        if trainModelsAndSave:
            model = neuralNetwork(train, epochs= epochs)
            model.save(modelDirectory + "leaveOut-" + str(participantNumber) + "-" + str(epochs) + "epochs.tflearn")
        else:
            model = neuralNetwork(train, train= False)
            model.load(modelDirectory + "leaveOut-" + str(participantNumber) + "-" + str(epochs) + "epochs.tflearn")

        # Append results to the end of the data frame
        results.loc[len(results)] = [participantNumber, False] + assessModelAccuracy(model, test)
        results.loc[len(results)] = [participantNumber, True] + assessModelAccuracy(model, test, shouldFilterOutMismatch= True)

    print(results)
    results.to_csv("./analyses/supervisoryCrossValidationResults-LeaveOut" + str(leaveOut) + "-" + str(epochs) + "epochs.csv")

# Human-Robot Teaming Generalizability - Train on Supervisory, test on Peer-Based
def supervisoryHumanRobot(epochs, leaveOut= [], trainModelsAndSave= True, respirationRate= True):
    features= ["meanIntensity", "stDevIntensity", "meanPitch", "stDevPitch", "stDevVoiceActivity", "meanVoiceActivity", "syllablesPerSecond", "filledPauses", "respirationRate"]

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
        modelDirectory = "./models/Supervisory_Human_Robot-LeaveOut" + str(leaveOut) + "/"

    trainDay1 = loadData(directory= day1Directory, audioFeatures= audioFeatures, respirationRate= includeRespirationRate, trimToRespirationLength= False)
    trainDay2 = loadData(directory= day2Directory, audioFeatures= audioFeatures, respirationRate= includeRespirationRate, trimToRespirationLength= False)
    train = pd.concat([trainDay1, trainDay2])

    test = loadData(directory= peerDirectory, audioFeatures= audioFeatures, trimToRespirationLength= False, respirationRate= includeRespirationRate, filter=False)

    print(train)

    if trainModelsAndSave:
        model = neuralNetwork(train, epochs= epochs)
        model.save(modelDirectory + "supervisoryHumanRobot" + str(epochs) + "epochs.tflearn")
    else:
        model = neuralNetwork(train, train= False)
        model.load(modelDirectory + "supervisoryHumanRobot" + str(epochs) + "epochs.tflearn")

    metrics = [[False] + assessModelAccuracy(model, test), [True] + assessModelAccuracy(model, test, shouldFilterOutMismatch= True)]

    # Append results to the end of the data frame
    results = pd.DataFrame(metrics, columns=["filtered", "samples", "coefficient", "RMSE", "actualMean", "actualStDev", "predMean", "predStDev"])

    print(results)
    results.to_csv("./analyses/supervisoryHumanRobot-LeaveOut" + str(leaveOut) + "-" + str(epochs) + "epochs.csv")

# Human-Robot Teaming Generalizability - Train on Peer-Based, test on Supervisory
def peerHumanRobot(epochs, leaveOut= [], trainModelsAndSave= True, respirationRate= True):
    features= ["meanIntensity", "stDevIntensity", "meanPitch", "stDevPitch", "stDevVoiceActivity", "meanVoiceActivity", "syllablesPerSecond", "filledPauses", "respirationRate"]

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
        modelDirectory = "./models/Peer_Human_Robot-LeaveOut" + str(leaveOut) + "/"
        # modelDirectory.replace("\'", "\\\'")

    # print(modelDirectory, os.path.exists(modelDirectory))

    train = loadData(directory= peerDirectory, audioFeatures= audioFeatures, trimToRespirationLength= False,
                     respirationRate= includeRespirationRate)

    testDay1 = loadData(directory= day1Directory, audioFeatures= audioFeatures,
                        respirationRate= includeRespirationRate, trimToRespirationLength= False, filter=False)
    testDay2 = loadData(directory= day2Directory, audioFeatures= audioFeatures,
                        respirationRate= includeRespirationRate, trimToRespirationLength= False, filter=False)
    test = pd.concat([testDay1, testDay2],  ignore_index= True)

    if trainModelsAndSave:
        model = neuralNetwork(train, epochs= epochs)
        model.save(modelDirectory + "peerHumanRobot" + str(epochs) + "epochs.tflearn")
    else:
        model = neuralNetwork(train, train= False)
        model.load(modelDirectory + "peerHumanRobot" + str(epochs) + "epochs.tflearn")

    metrics = [[False] + assessModelAccuracy(model, test), [True] + assessModelAccuracy(model, test, shouldFilterOutMismatch= True)]

    # Append results to the end of the data frame
    results = pd.DataFrame(metrics, columns=["filtered", "samples", "coefficient", "RMSE", "actualMean", "actualStDev", "predMean", "predStDev"])

    print(results)
    results.to_csv("./analyses/peerHumanRobot-LeaveOut" + str(leaveOut) + "-" + str(epochs) + "epochs.csv")

# Real-time evaluaiton sanity check - Train on Supervisory, test on Real-Time
def realTimeSanityCheck(epochs, leaveOut= [], trainModelsAndSave= True, respirationRate= True):
    features= ["meanIntensity", "stDevIntensity", "meanPitch", "stDevPitch", "stDevVoiceActivity", "meanVoiceActivity", "syllablesPerSecond", "filledPauses", "respirationRate"]

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
        modelDirectory = "./models/Real_Time_Sanity-LeaveOut" + str(leaveOut) + "/"

    test = loadData(directory= realTimeDirectory, audioFeatures= audioFeatures, trimToRespirationLength= False, respirationRate= includeRespirationRate, filter=False)

    trainDay1 = loadData(directory= day1Directory, audioFeatures= audioFeatures, respirationRate= includeRespirationRate, trimToRespirationLength= False)
    trainDay2 = loadData(directory= day2Directory, audioFeatures= audioFeatures, respirationRate= includeRespirationRate, trimToRespirationLength= False)
    train = pd.concat([trainDay1, trainDay2])

    print(test)

    if trainModelsAndSave:
        model = neuralNetwork(train, epochs= epochs)
        model.save(modelDirectory + "realTimeSanityCheck" + str(epochs) + "epochs.tflearn")
    else:
        model = neuralNetwork(train, train= False)
        model.load(modelDirectory + "realTimeSanityCheck" + str(epochs) + "epochs.tflearn")

    metrics = [[False] + assessModelAccuracy(model, test), [True] + assessModelAccuracy(model, test, shouldFilterOutMismatch= True)]

    # Append results to the end of the data frame
    results = pd.DataFrame(metrics, columns=["filtered", "samples", "coefficient", "RMSE", "actualMean", "actualStDev", "predMean", "predStDev"])

    print(results)
    results.to_csv("./analyses/realTimeSanityCheck-LeaveOut" + str(leaveOut) + "-" + str(epochs) + "epochs.csv")


# Real-time window size evaluation
def realTimeWindowSizeEvaluation(epochs, trainModelsAndSave= True):
    audioFeatures= ["meanIntensity", "stDevIntensity", "meanPitch", "stDevPitch", "stDevVoiceActivity", "meanVoiceActivity", "syllablesPerSecond", "filledPauses"]

    directories = {
                   1:"Real_Time-1_second_window",
                   5:"Real_Time-5_second_window",
                   10:"Real_Time-10_second_window",
                   15:"Real_Time-15_second_window",
                   30:"Real_Time-30_second_window",
                   60:"Real_Time-60_second_window"}

    results = pd.DataFrame(columns=["windowSize", "participant", "filtered", "samples", "coefficient", "RMSE", "actualMean", "actualStDev", "predMean", "predStDev"])

    for windowSize, directory in directories.items():
        modelDirectory = "./models/Real_Time_Window_Size-" + str(windowSize) + "_seconds/"
        featureDirectory = "./training/" + directory + "/"

        for participantNumber in range(1, 31):
            print("Participant", participantNumber, "windowSize:", windowSize)

            participantPath = ""

            filePathsInDirectory = list(sorted(glob.iglob(featureDirectory + "features/*.csv")))

            featurePaths = [path for path in filePathsInDirectory if (os.path.basename(path)[:-4][0] == 'p'
                                                                      and os.path.basename(path)[:-4][1:].isdigit())]

            for path in featurePaths:
                if str(participantNumber) == os.path.basename(path).split('_')[0][1:][:-4]:
                    participantPath = path

            if participantPath != "": # some participants are missing from the dataset

                featurePaths.remove(participantPath)

                train = loadData(trainingFiles=featurePaths, audioFeatures= audioFeatures,
                                 respirationRate= False, trimToRespirationLength= False)
                test = loadData(trainingFiles=[participantPath], audioFeatures= audioFeatures,
                                respirationRate= False, trimToRespirationLength= False, filter=False)

            if trainModelsAndSave:
                model = neuralNetwork(train, epochs= epochs)
                model.save(modelDirectory + "leaveOut-" + str(participantNumber) + "-" + str(epochs) + "epochs.tflearn")
            else:
                model = neuralNetwork(train, train= False)
                model.load(modelDirectory + "leaveOut-" + str(participantNumber) + "-" + str(epochs) + "epochs.tflearn")

            # Append results to the end of the data frame
            results.loc[len(results)] = [windowSize, participantNumber, False] + assessModelAccuracy(model, test)
            results.loc[len(results)] = [windowSize, participantNumber, True] + assessModelAccuracy(model, test, shouldFilterOutMismatch= True)

            print(results)

    results.to_csv("./analyses/realTimeWindowSizeEvaluation-" + str(epochs) + "epochs.csv")


def main():
    # supervisoryRealWorld(50, trainModelsAndSave= False)
    # supervisoryRealWorld(50, trainModelsAndSave= False, leaveOut= ["filledPauses"])
    # supervisoryRealWorld(50, trainModelsAndSave= True, leaveOut= ["respirationRate"])

    # supervisoryLeaveOneOutCrossValidation(50, trainModelsAndSave= True)
    # supervisoryLeaveOneOutCrossValidation(50, trainModelsAndSave= True, leaveOut= ["filledPauses"])
    # supervisoryLeaveOneOutCrossValidation(50, trainModelsAndSave= True, leaveOut= ["respirationRate"])

    # supervisoryHumanRobot(100, trainModelsAndSave= True, leaveOut= ["respirationRate"])
    # supervisoryHumanRobot(100, trainModelsAndSave= True, leaveOut= ["respirationRate", "filledPauses"])

    # peerHumanRobot(100, trainModelsAndSave= True, leaveOut= ["respirationRate"])
    # peerHumanRobot(100, trainModelsAndSave= True, leaveOut= ["respirationRate", "filledPauses"])

    # realTimeSanityCheck(50, trainModelsAndSave= True)
    # realTimeSanityCheck(50, trainModelsAndSave= True, leaveOut= ["filledPauses"])
    # realTimeSanityCheck(50, trainModelsAndSave= True, leaveOut= ["respirationRate"])

    realTimeWindowSizeEvaluation(50)


if __name__ == "__main__":
    main()
