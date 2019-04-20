#
# Created on Nov 12, 2018
#
# @author: Julian Fortune
#
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tflearn
import glob

np.set_printoptions(threshold=np.nan)

def loadData():
    ulLabelPath = "./labels/ul.npy"
    nlLabelPath = "./labels/nl.npy"
    olLabelPath = "./labels/ol.npy"

    ulLabels = np.load(ulLabelPath)
    nlLabels = np.load(nlLabelPath)
    olLabels = np.load(olLabelPath)

    labels = np.empty(0)
    inputs = np.asarray([[],[],[],[],[],[],[]])

    dir = "./features/*.npy"

    for path in sorted(glob.iglob(dir),reverse=False):
        currentInput = np.load(path)[1:] # [speech rate, mean pitch, stDev pitch, mean intensity, stDev intensity, mean voice activity, stDev voice activity]

        condition = path[:-4][-2:]
        if condition == "ul":
            currentLabels = ulLabels
        elif condition == "nl":
            currentLabels = nlLabels
        else:
            currentLabels = olLabels

        if len(currentInput[0]) > len(currentLabels):
            offsetSize = len(currentInput[0]) - len(currentLabels)
            offset = np.zeros(offsetSize)
            currentLabels = np.append(currentLabels, offset)

        if len(currentLabels) == len(currentInput[0]):
            labels = np.append(labels, currentLabels)
            inputs = np.append(inputs, currentInput, axis=-1)

    # Random solutions I found on the internet
    inputs = np.swapaxes(inputs,0,1)
    labels = np.reshape(labels, (-1, 1))

    print(len(inputs), len(labels))
    print(inputs[0], labels[0])
    print(inputs[12000], labels[12000])
    print(inputs[1650], labels[1650])


    return inputs, labels

def trainNetwork(inputs, labels):
    # Neural network characteristics
    input_neurons = 7
    hidden_neurons = 128
    output_neurons = 1

    n_epoch = 50 #try 20 and 50

    name = "SpeechWorkload"

    with tf.device('/gpu:0'):
        # Set up
        tf.reset_default_graph()
        tflearn.init_graph()

        #
        inputs, labels = tflearn.data_utils.shuffle(inputs, labels)

        # Input layer
        net = tflearn.input_data(shape=[None,input_neurons])

        # Hidden layers
        net = tflearn.fully_connected(net, hidden_neurons, bias=True, activation = 'relu')
        net = tflearn.fully_connected(net, hidden_neurons, bias=True, activation = 'relu')
        net = tflearn.fully_connected(net, hidden_neurons, bias=True, activation = 'relu')

        # Output layer
        net = tflearn.fully_connected(net, output_neurons)

        # Confused ???
        net = tflearn.regression(net, optimizer='Adam', learning_rate=0.001,  loss='mean_square' ,metric = 'R2', restore=True, batch_size=64)

        # Confused ???
        model = tflearn.DNN(net, tensorboard_verbose=0)

        # Fit the data, `validation_set=` sets asside a proportion of the data to validate with
        model.fit(inputs, labels, n_epoch=n_epoch, validation_set=0.10, show_metric=True)

        predictions = model.predict(inputs)
        print(predictions)

    model.save(name + '.tflearn')

def main():
    inputs, labels = loadData()
    trainNetwork(inputs, labels)

if __name__ == "__main__":
    main()
