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



def main():
    createValidationSetFromParticipants()

main()
