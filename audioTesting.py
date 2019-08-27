#
# Created on Apr 23, 2019
#
# @author: Julian Fortune
# @Description: Interface for extracting feature arrays.
#

import os, glob, sys # file io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from speechLibrary import speechAnalysis, featureModule, audioModule

def testUnBiasing():
    realTimeDirectory = "../media/Jamison_Evaluations/Real_Time_Evaluation/Audio/"

    for filePath in sorted(glob.iglob(realTimeDirectory + "*.wav")):
        name = os.path.basename(filePath)

        print(name)

        # Read in the file
        audio = audioModule.Audio(filePath)

        plotAudio(audio= audio, name= name, samples= 80000)

        audio.unBias()

        plotAudio(audio= audio, name= name, samples= 80000)


        audio = audioModule.Audio(filePath)
        if audio.numberOfChannels > 1:
            audio.makeMono()

        plotAudio(audio= audio, name= name, samples= 80000)

        audio.unBias()

        plotAudio(audio= audio, name= name, samples= 80000)

        # plotAudio(audio= audio, name= name)

def plotAudio(audio, name, samples= None):
    length = samples
    if not samples:
        length = audio.data.shape[0]

    if audio.numberOfChannels == 2:
        signalTimes = np.arange(0, len(audio.data)/audio.sampleRate, 1 / audio.sampleRate)[:length]

        fig, axs = plt.subplots(2, figsize=[16, 8])
        axs[0].plot(signalTimes, audio.data[:length, 0])
        axs[1].plot(signalTimes, audio.data[:length, 1])

        plt.title(name)
        plt.show()
    else:
        analyzer = speechAnalysis.SpeechAnalyzer()
        signalTimes = np.arange(0, len(audio.data)/audio.sampleRate, 1 / audio.sampleRate)[:length]

        plt.figure(figsize=[16, 8])
        plt.plot(signalTimes, audio.data[:length])
        # plt.plot(energyTimes, energy)
        plt.title(name)
        plt.show()



def main():
    testUnBiasing()

main()
