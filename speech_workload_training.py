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

# np.set_printoptions(threshold=sys.maxsize)

def loadData(directory, trainingFiles=[], inputFeaturesToDiscard=[], filterWithVoiceActivity=True):
    # Always discard time
    if "time" not in inputFeaturesToDiscard:
        inputFeaturesToDiscard.append("time")

    ulLabelPath = directory + "labels/ul.npy"
    nlLabelPath = directory + "labels/nl.npy"
    olLabelPath = directory + "labels/ol.npy"

    ulLabels = np.load(ulLabelPath)
    nlLabels = np.load(nlLabelPath)
    olLabels = np.load(olLabelPath)

    inputFeatureNames = []
    with open(directory + 'features/labels.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            inputFeatureNames = row

    # Ensure features to discard actually exist
    for feature in inputFeaturesToDiscard:
        assert feature in inputFeatureNames, "Invalid feature to discard."

    # Sort the features to discard based on the reverse of their positioning in the feature set.
    inputFeaturesToDiscard.sort(key=lambda feature: inputFeatureNames.index(feature), reverse=True)

    numberOfInputs = len(inputFeatureNames) - len(inputFeaturesToDiscard)

    labels = np.empty(0)
    inputs = np.asarray([[]] * numberOfInputs)

    for path in sorted(glob.iglob(directory + "features/*.npy")):
        # Check if the file should be used in the training set, if a subset of
        # files is specified.
        if (not trainingFiles) or (path in trainingFiles):
            currentInput = np.load(path)

            # Remove features that should be discarded.
            for feature in inputFeaturesToDiscard:
                currentInput = np.delete(currentInput, inputFeatureNames.index(feature), 0)

            condition = path[:-4][-2:]
            assert condition in ["ul", "nl", "ol"]

            if condition == "ul":
                currentLabels = ulLabels
            elif condition == "nl":
                currentLabels = nlLabels
            else:
                currentLabels = olLabels

            # Add extra zeros to the labels if inputs run over the length
            if len(currentInput[0]) > len(currentLabels):
                offsetSize = len(currentInput[0]) - len(currentLabels)
                offset = np.zeros(offsetSize)
                currentLabels = np.append(currentLabels, offset)

            if len(currentLabels) == len(currentInput[0]):
                labels = np.append(labels, currentLabels)
                inputs = np.append(inputs, currentInput, axis=-1)
            else:
                print("WARNING: Shapes of labels and inputs do not match.", currentLabels.shape, currentInput.shape)

    assert not np.isnan(inputs).any(), "Invalid values in inputs."

    # Rearrange the feature data to have all feature values together for each
    # time instance instead of lists of features.
    inputs = np.swapaxes(inputs,0,1)

    labels = np.reshape(labels, (-1, 1))

    for feature in inputFeaturesToDiscard:
        inputFeatureNames.remove(feature)

    # print("Using", inputFeatureNames)

    if filterWithVoiceActivity:
        acceptableTrainingSamples = (inputs[:,inputFeatureNames.index("meanVoiceActivity")] > 0.1) & (labels[:,0] > 0)
        inputs = inputs[acceptableTrainingSamples]
        labels = labels[acceptableTrainingSamples]

    dataFrame = pd.DataFrame(inputs, columns= inputFeatureNames)
    dataFrame['speechWorkload'] = labels

    return dataFrame

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

    # graphData = pd.concat([results, data.drop(columns=['speechWorkload'])], axis=1)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(results)

    correlationCoefficient = stats.pearsonr(actual, predictions)[0]

    rmse = np.sqrt(np.mean((predictions - actual)**2, axis=0, keepdims=True))[0]

    actualMean = np.mean(actual)
    actualStandardDeviation = np.std(actual)
    predictionsMean = np.mean(predictions)
    predictionsStandardDeviation = np.std(predictions)

    print(correlationCoefficient, rmse, actualMean, actualStandardDeviation, predictionsMean, predictionsStandardDeviation)

    # results.plot()
    # plt.show()

def supervisoryLeaveOneOutCrossValidation():
    directory = "./training/Supervisory_Evaluation_Day_1/"

    results = pd.DataFrame(columns=["file"])

    trainModelsAndSave = True
    participants = []

    for participantNumber in range(1, 31):
        print("Participant", participantNumber)

        participantPaths = []

        for path in sorted(glob.iglob(directory + "features/*.npy")):

            if str(participantNumber) == os.path.basename(path).split('_')[0][1:]:
                participantPaths.append(path)

        featurePaths = sorted(glob.iglob(directory + "features/*.npy"))

        for path in participantPaths:
            featurePaths.remove(path)

        train = loadData(directory, trainingFiles=featurePaths)
        test = loadData(directory, trainingFiles=participantPaths, filterWithVoiceActivity=False)

        if trainModelsAndSave:
            model = neuralNetwork(train, directory)
            model.save(directory + "models/leaveOut-" + os.path.basename(path) + ".tflearn")
        else:
            model.load(directory + "models/leaveOut-" + os.path.basename(path) + ".tflearn")
            metrics = assessModelAccuracy(model, test)

def main():
    supervisoryLeaveOneOutCrossValidation()

if __name__ == "__main__":
    main()
