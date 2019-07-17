#
# Created on Nov 12, 2018
#
# @author: Julian Fortune
#

import glob, sys, csv

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tflearn

np.set_printoptions(threshold=sys.maxsize)

def loadData(directory, inputFeaturesToDiscard=[]):
    # Always discard time
    if "time" not in inputFeaturesToDiscard:
        inputFeaturesToDiscard.append("time")

    ulLabelPath = "./labels/ul.npy"
    nlLabelPath = "./labels/nl.npy"
    olLabelPath = "./labels/ol.npy"

    ulLabels = np.load(ulLabelPath)
    nlLabels = np.load(nlLabelPath)
    olLabels = np.load(olLabelPath)

    inputFeatureNames = []
    with open(directory + 'labels.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            inputFeatureNames = row

    for feature in inputFeaturesToDiscard:
        assert feature in inputFeatureNames, "Invalid feature to discard."

    numberOfInputs = len(inputFeatureNames) - len(inputFeaturesToDiscard)

    labels = np.empty(0)
    inputs = np.asarray([[]] * numberOfInputs)

    for path in sorted(glob.iglob(directory + "*.npy"),reverse=False):
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

    print(len(inputs), len(labels))
    print(inputs.shape)


    return inputs, labels

def trainNetwork(inputs, labels):
    # Neural network characteristics
    input_neurons = inputs.shape[1] # Size in the second dimension
    hidden_neurons = 128
    output_neurons = 1

    n_epoch = 50 #try 20 and 50

    name = "SpeechWorkload"

    with tf.device('/gpu:0'):
        # Set up
        tf.reset_default_graph()
        tflearn.init_graph()

        # Shuffle data
        inputs, labels = tflearn.data_utils.shuffle(inputs, labels)

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

        print(net)

        # Create the model from the network
        model = tflearn.DNN(net, tensorboard_verbose=0)

        # Fit the data, `validation_set=` sets asside a proportion of the data to validate with
        model.fit(inputs, labels, n_epoch=n_epoch, validation_set=0.10, show_metric=True)

    model.save(name + '.tflearn')


def main():
    inputs, labels = loadData("./features/")
    trainNetwork(inputs, labels)

if __name__ == "__main__":
    main()
