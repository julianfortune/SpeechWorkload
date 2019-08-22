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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pydub import AudioSegment

from speechLibrary import featureModule, speechAnalysis, audioModule

# np.set_printoptions(threshold=sys.maxsize)


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
    praatSyllables =                [1189, 1099, 1108, 1012,  955, 1079, 1109,  983, 1185,  903]
    ourAlgorithmCorrectDetections = [ 639,  565,  620,  534,  453,  587,  555,  580,  579,  486]
    ourAlgorithmFalseAlarms =       [ 272,  252,  222,  240,  193,  276,  206,  230,  253,  250]

    graphValidationResults(praatSyllables,
                           ourAlgorithmCorrectDetections,
                           ourAlgorithmFalseAlarms,
                           validationAlgorithmLabel= "rVAD",
                           algorithmLabel= "Algorithm",
                           yLabel= "Syllables detected",
                           xLabel= "Validation Set",
                           title= "Syllable Algorithm Comparison")

def graphVoiceActivityResults():
    rVADVoiceActivityInstances = [654, 588, 609, 585, 601, 611, 605, 603, 602, 629]
    ourAlgorithmVoiceActivity =  [271, 224, 249, 265, 262, 237, 272, 233, 250, 268]
    ourAlgorithmFalseAlarms =    [ 77,  73,  66,  53,  41,  49,  69,  70,  61,  45]

    graphValidationResults(rVADVoiceActivityInstances,
                           ourAlgorithmVoiceActivity,
                           ourAlgorithmFalseAlarms,
                           validationAlgorithmLabel= "rVAD",
                           algorithmLabel= "Algorithm",
                           yLabel= "Positive Voice Activity Classifications",
                           xLabel= "Validation Set",
                           title= "Voice Activity Algorithm Comparison")

def graphFilledPausesResults():
    actualFilledPausesCount =       [ 50, 78,  7, 48,  2, 29,  2,  8,  0,   0,  0]
    ourAlgorithmFilledPausesCount = [ 36,  9,  6, 29,  2, 13,  0,  1,  0,   0,  0]
    ourAlgorithmFalseAlarms =       [  0,  0,  0,  1, 17,  2,  1,  0,  6,  32,  1]

    graphValidationResults(actualFilledPausesCount,
                           # np.add(ourAlgorithmFilledPausesCount, ourAlgorithmFalseAlarms).astype(int),
                           # np.zeros(len(ourAlgorithmFalseAlarms)).astype(int),
                           ourAlgorithmFilledPausesCount,
                           ourAlgorithmFalseAlarms,
                           validationAlgorithmLabel= "Actual",
                           algorithmLabel= "Algorithm",
                           yLabel= "Filled Pauses (Detected)",
                           xLabel= "Participant",
                           title= "Filled Pauses Algorithm Comparison",
                           tickLabels=["102", "103", "104", "106", "107", "108", "109", "110", "111", "113", "114"],
                           correctDetectionShift=2)

def graphVoiceActivityComparison():
    filePath = "../validation/results/voiceActivity.csv"
    outputDirectoryPath = "../validation/results/voiceActivity/"
    results = pd.read_csv(filePath)

    validationColor = "mediumaquamarine"
    ourColor = "mediumpurple"

    plt.rc('font',**{'family':'serif','serif':['Palatino']})

    for index, line in results.iterrows():
        fileName = "p" + "_".join([str(line["participant"]), str(line["condition"]), line["times"]])
        outputPath = outputDirectoryPath + str(line["validationSet"]) + "/" + fileName + ".png"

        rVADValues = [float(element) for element in line[["rVAD"]].values[0].strip('][').split(', ')]
        ourAlgorithmValues = [float(element) for element in line[["ourAlgorithm"]].values[0].strip('][').split(', ')]

        rVADValues = [0 if element == 0 else 1 for element in rVADValues]
        ourAlgorithmValues = [0 if element == 0 else -1 for element in ourAlgorithmValues]

        rVAD = pd.DataFrame(rVADValues + [0], columns= ["rVAD"])
        ourAlgorithm = pd.DataFrame(ourAlgorithmValues + [0], columns= ["Our Algorithm"])

        plt.figure(figsize=(7,3))

        plt.step(list(rVAD.index), rVAD[["rVAD"]].values[:,0], drawstyle="steps", color=validationColor, label="Praat", where="post")
        plt.step(list(ourAlgorithm.index), ourAlgorithm[["Our Algorithm"]].values[:,0], drawstyle="steps", color=ourColor, label="Our Algorithm", where="post")
        plt.plot(list(range(0, len(list(rVAD.index)) + 1)), [0] * (len(list(rVAD.index)) + 1), drawstyle="steps", color="black")

        plt.fill_between(list(rVAD.index), rVAD[["rVAD"]].values[:,0], step="post", color=validationColor)
        plt.fill_between(list(ourAlgorithm.index), ourAlgorithm[["Our Algorithm"]].values[:,0], step="post", color=ourColor)

        plt.title(fileName)
        plt.ylim((-3, 3))
        plt.xlim((0,30))
        plt.box(on=None)
        plt.subplots_adjust(top= 0.85, bottom= 0.25)
        plt.yticks([])

        patch1 = mpatches.Patch(color=validationColor)
        patch2 = mpatches.Patch(color=ourColor)

        patches = (patch1, patch2)

        plt.legend(patches, ["rVAD", "Ours"], loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2, )

        plt.savefig(outputPath)
        plt.close()

def graphSyllablesComparison():
    filePath = "../validation/results/syllables.csv"
    outputDirectoryPath = "../validation/results/syllables/"
    results = pd.read_csv(filePath)

    validationColor = "mediumaquamarine"
    ourColor = "mediumpurple"

    plt.rc('font',**{'family':'serif','serif':['Palatino']})

    for index, line in results.iterrows():
        fileName = "p" + "_".join([str(line["participant"]), str(line["condition"]), line["times"]])
        outputPath = outputDirectoryPath + str(line["validationSet"]) + "/" + fileName + ".png"

        praatValues = line[["rVAD"]].values[0].strip('][').split(', ')
        if not praatValues[0]:
            praatValues = []

        ourAlgorithmValues = line[["ourAlgorithm"]].values[0].strip('][').split(', ')
        if not ourAlgorithmValues[0]:
            ourAlgorithmValues = []

        praatValues = [float(element) for element in praatValues]
        ourAlgorithmValues = [float(element) for element in ourAlgorithmValues]

        plt.figure(figsize=(11,3))
        plt.plot(list(range(0, 30 + 1)), [0] * (30 + 1), drawstyle="steps", color="black")

        for x_pos in praatValues:
            plt.vlines(x_pos, ymin=0, ymax=1, color=validationColor)

        for x_pos in ourAlgorithmValues:
            plt.vlines(x_pos, ymin=-1, ymax=0, color=ourColor)

        plt.title(fileName)
        plt.ylim((-1.5, 1.5))
        plt.xlim((0,30))
        plt.box(on=None)
        plt.subplots_adjust(top= 0.85, bottom= 0.25)
        plt.yticks([])

        patch1 = mpatches.Patch(color=validationColor)
        patch2 = mpatches.Patch(color=ourColor)

        patches = (patch1, patch2)

        plt.legend(patches, ["Praat", "Ours"], loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2, )

        plt.savefig(outputPath)
        plt.close()



def main():
    graphVoiceActivityComparison()
    print("Done w voice activity")
    graphSyllablesComparison()
    print("Done w syllables")

main()
