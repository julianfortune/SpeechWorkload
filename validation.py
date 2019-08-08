#
# Created on July 11, 2019
#
# @author: Julian Fortune
# @Description: Functions for validating speech characteristics algorithms.
#

import sys, time, glob, os
import wavio
import csv

import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment

import random

from speechLibrary import featureModule, speechAnalysis, audioModule

np.set_printoptions(threshold=sys.maxsize)

audioDirectory = "../media/validation_participant_audio/"

def createMultipleValidationSets():
    participantDirectoryPath = "../media/Participant_Audio/"
    outputDirectoryPath = "../media/validation_testing/"
    numberOfSets = 10
    segmentLengthInSeconds = 30

    for setNumber in range(numberOfSets):
        setPath = outputDirectoryPath + str(setNumber + 1) + "/"
        os.mkdir(setPath)
        createValidationSetFromParticipants(participantDirectory= participantDirectoryPath,
                                            outputDir= setPath,
                                            segmentLengthInSeconds= segmentLengthInSeconds)

# | Makes a 30-second segment from each audio file (30 participants x 3 conditions)
def createValidationSetFromParticipants(participantDirectory, outputDir, segmentLengthInSeconds= 30):
    segmentLength = segmentLengthInSeconds * 1000 # In milliseconds

    for filePath in  sorted(glob.iglob(participantDirectory + "*.wav")):
        name = os.path.basename(filePath)[:-4]

        audio = AudioSegment.from_wav(filePath)

        audioObject = audioModule.Audio(filePath=filePath)
        audioObject.makeMono()

        length = int(len(audioObject.data) / audioObject.sampleRate * 1000)
        segmentStartRange = length - segmentLength

        start = random.randrange(segmentStartRange)
        end = start + segmentLength

        segment = audio[start:end]

        outputPath = outputDir + name + "_" + str(round(start/1000, 2)) + "-" + str(round(end/1000, 2))

        print(outputPath)

        segment.export(outputPath + ".wav", format="wav")

def graphValidationResults(validationAlgorithmValues, ourAlgorithmCorrectDetections, ourAlgorithmFalseAlarms, validationAlgorithmLabel, algorithmLabel, yLabel, xLabel, title, tickLabels=None, correctDetectionShift=15):
    x = np.linspace(0, len(validationAlgorithmValues) - 1, len(validationAlgorithmValues))

    width = .3
    spacing = 0.05

    hShift = -0.01

    plt.rc('font',**{'family':'serif','serif':['Palatino']})

    figure = plt.figure(figsize=(14,5))

    plt.bar(x, validationAlgorithmValues, width, label=validationAlgorithmLabel, color="white", edgecolor="black")
    plt.bar(x + width + spacing, ourAlgorithmCorrectDetections, width, label=algorithmLabel, color="lightgrey", edgecolor="black")
    plt.bar(x + width + spacing, ourAlgorithmFalseAlarms, width, label=algorithmLabel + " False Alarms", bottom=ourAlgorithmCorrectDetections, hatch="////", color="lightgrey", edgecolor="black")

    if tickLabels:
        plt.xticks(x + (width + spacing) / 2, tickLabels)
    else:
        plt.xticks(x + (width + spacing) / 2, x.astype(int) + 1)

    for xValue, yValue in enumerate(validationAlgorithmValues):
        plt.text(xValue + hShift, yValue, " " + str(yValue),
                 color= "black", va= "bottom", ha= "center")

    for xValue, yValue in enumerate(ourAlgorithmFalseAlarms):
        vShift = ourAlgorithmCorrectDetections[int(xValue)]

        plt.text(xValue + hShift + spacing + width, yValue + vShift, " " + str(yValue + vShift),
                 color='black', va='bottom', ha='center')

    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=5)
    plt.margins(0.05, 0.1)
    plt.subplots_adjust(bottom=0.2)
    plt.title(title)

    plt.show()

def graphSyllableResults():
    praatSyllables =                [1189, 1185, 1099, 1108, 1012,  955, 1079, 1109,  983,  903]
    ourAlgorithmCorrectDetections = [ 802,  735,  712,  744,  693,  573,  761,  703,  710,  630]
    ourAlgorithmFalseAlarms =       [ 109,   97,  105,   98,   81,   73,  102,   58,  100,  106]

    graphValidationResults(praatSyllables,
                           ourAlgorithmCorrectDetections,
                           ourAlgorithmFalseAlarms,
                           validationAlgorithmLabel= "PRAAT Script",
                           algorithmLabel= "Algorithm",
                           yLabel= "Syllables detected",
                           xLabel= "Validation Set",
                           title= "Syllable Algorithm Comparison")

def graphVoiceActivityResults():
    rVADVoiceActivityInstances = [ 654, 609, 585, 601, 611, 605, 603, 602, 629, 588]
    ourAlgorithmVoiceActivity =  [ 271, 249, 265, 262, 237, 272, 233, 250, 268, 224]
    ourAlgorithmFalseAlarms =    [  77,  66,  53,  41,  49,  69,  70,  61,  45,  73]

    graphValidationResults(rVADVoiceActivityInstances,
                           ourAlgorithmVoiceActivity,
                           ourAlgorithmFalseAlarms,
                           validationAlgorithmLabel= "rVAD",
                           algorithmLabel= "Algorithm",
                           yLabel= "Positive Voice Activity Classifications",
                           xLabel= "Validation Set",
                           title= "Voice Activity Algorithm Comparison")

def graphFilledPausesResults():
    actualFilledPausesCount =       [ 50, 72,  7, 48,  1, 27,  0,  8,  0,   0,  0]
    ourAlgorithmFilledPausesCount = [ 36,  9,  6, 29,  1, 13,  0,  1,  0,   0,  0]
    ourAlgorithmFalseAlarms =       [  0,  0,  0,  1, 18,  2,  1,  0,  6,  32,  1]

    graphValidationResults(actualFilledPausesCount,
                           ourAlgorithmFilledPausesCount,
                           ourAlgorithmFalseAlarms,
                           validationAlgorithmLabel= "Actual",
                           algorithmLabel= "Algorithm",
                           yLabel= "Filled Pauses (Detected)",
                           xLabel= "Participant",
                           title= "Filled Pauses Algorithm Comparison",
                           tickLabels=["102", "103", "104", "106", "107", "108", "109", "110", "111", "113", "114"],
                           correctDetectionShift=2)



def main():
    # graphSyllableResults()
    # graphVoiceActivityResults()
    graphFilledPausesResults()

main()
