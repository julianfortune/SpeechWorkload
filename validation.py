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

# | Makes a 30-second segment from each audio file (30 participants x 3 conditions)
def createValidationSetFromParticipants():
    audioDirectory = "../media/Participant_Audio/*.wav"
    outputDir = "../media/validation_participant_audio/"

    segmentLength = 30 * 1000 # In milliseconds

    for filePath in glob.iglob(audioDirectory):
        name = os.path.basename(filePath)[:-4]

        participant = name.split("_")[0]
        condition = name.split("_")[1]

        print(participant, condition)

        audio = AudioSegment.from_wav(filePath)
        audioObject = audioModule.Audio(filePath=filePath)
        audioObject.makeMono()

        length = int(len(audioObject.data) / audioObject.sampleRate * 1000)
        range = length - segmentLength

        start = random.randrange(range)
        end = start + segmentLength

        segment = audio[start:end]

        outputPath = outputDir + name + "_" + str(round(start/1000, 2)) + "-" + str(round(end/1000, 2))

        print(outputPath)

        segment.export(outputPath + ".wav", format="wav")



def main():
    createValidationSetFromParticipants()

main()
