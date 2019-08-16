#
# Created on Nov 12, 2018
#
# @author: Julian Fortune
#

import glob, sys, csv, os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tflearn
import pandas as pd
from scipy import stats



def loadData(directory, trainingFiles= None, filter= True, threshold= 0.1, audioFeatures= ["meanIntensity", "stDevIntensity", "meanPitch", "stDevPitch", "stDevVoiceActivity", "meanVoiceActivity", "syllablesPerSecond", "filledPauses"], respirationRate= True):
    data = pd.DataFrame(columns= audioFeatures + (["respirationRate"] if respirationRate else None) + ["speechWorkload"])

    for path in sorted(glob.iglob(directory + "features/*.csv")):
        # Check if the file should be used in the training set, if a subset of
        # files is specified.
        if (not trainingFiles) or (path in trainingFiles):
            name = os.path.basename(path)[:-4]
            currentData = pd.read_csv(path, index_col= 0)
            currentLabels = pd.read_csv(directory + "labels/" + name + ".csv", index_col= 0)

            if len(currentLabels) == len(currentData.index):
                # Add the speech workload values
                currentData['speechWorkload'] = currentLabels

                if respirationRate:
                    respirationRateData = currentLabels = pd.read_csv(directory + "physiological/" + name + ".csv", index_col= 0)
                    currentData["respirationRate"] = respirationRateData
                    currentData = currentData.dropna()

                data = data.append(currentData[data.columns], ignore_index= True)
            else:
                print("WARNING: Shapes of labels and inputs do not match.", currentLabels.shape, currentData.shape)

    print("Using", list(data.columns))

    if filter:
        data = data[(data['meanVoiceActivity'] > threshold) & (data['speechWorkload'] > 0)]

    return data

def neuralNetwork(data, directory, train= True):
    inputs = data.drop(columns=['speechWorkload']).to_numpy()
    labels = np.reshape(data['speechWorkload'].to_numpy(), (-1, 1))

    # Shuffle data
    inputs, labels = tflearn.data_utils.shuffle(inputs, labels)

    # Neural network characteristics
    input_neurons = inputs.shape[1] # Size in the second dimension
    hidden_neurons = 256
    output_neurons = 1

    n_epoch = 50 #try 20 - 2000

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
            model.fit(inputs, labels, n_epoch=n_epoch, validation_set=0.10, show_metric=True)

        return model

def assessModelAccuracy(model, data):
    predictions = model.predict(data.drop(columns=['speechWorkload']).to_numpy())[:,0]
    predictions[data.meanVoiceActivity < 0.1] = 0

    actual = data.speechWorkload.to_numpy()

    results = pd.DataFrame()
    results['predictions'] = predictions
    results['actual'] = data.speechWorkload

    graphData = pd.concat([results, data.drop(columns=['speechWorkload'])], axis=1)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(results)

    correlationCoefficient = stats.pearsonr(actual, predictions)[0]

    rmse = np.sqrt(np.mean((predictions - actual)**2, axis=0, keepdims=True))[0]

    actualMean = np.mean(actual)
    actualStandardDeviation = np.std(actual)
    predictionsMean = np.mean(predictions)
    predictionsStandardDeviation = np.std(predictions)

    # pd.concat([results, data.meanIntensity, data.meanVoiceActivity], axis=1).plot()
    # plt.show()

    return [correlationCoefficient, rmse, actualMean, actualStandardDeviation, predictionsMean, predictionsStandardDeviation]

def supervisoryLeaveOneOutCrossValidation():
    directory = "./training/Supervisory_Evaluation_Day_2/"

    results = pd.DataFrame(columns=["participant", "coefficient", "RMSE", "actualMean", "actualStDev", "predMean", "predStDev"])

    trainModelsAndSave = False
    participants = []

    for participantNumber in range(1, 31):
        print("Participant", participantNumber)

        participantPaths = []

        for path in sorted(glob.iglob(directory + "features/*.csv")):

            if str(participantNumber) == os.path.basename(path).split('_')[0][1:]:
                participantPaths.append(path)

        featurePaths = sorted(glob.iglob(directory + "features/*.csv"))

        for path in participantPaths:
            featurePaths.remove(path)

        train = loadData(directory, trainingFiles=featurePaths)
        test = loadData(directory, trainingFiles=participantPaths, filter=False)

        # if trainModelsAndSave:
        model = neuralNetwork(train, directory)
        model.save(directory + "models/leaveOut-" + str(participantNumber) + ".tflearn")
        # else:
        # model = neuralNetwork(train, directory, train= False)
        # model.load(directory + "models/leaveOut-" + str(participantNumber) + ".tflearn")
        metrics = assessModelAccuracy(model, test)

        # Append results to the end of the data frame
        results.loc[len(results)] = [participantNumber] + metrics

    print(results)

def testPhysio():
    directory = "./training/Supervisory_Evaluation_Day_2/"

    train = loadData(directory)


def main():
    testPhysio()

if __name__ == "__main__":
    main()
