import numpy as np
import matplotlib.pyplot as plt

import sys, time, glob, os

import wavio
import csv

from pydub import AudioSegment

from speechLibrary import featureModule, speechAnalysis, audioModule

np.set_printoptions(threshold=sys.maxsize)

# Parameters of the features
utteranceWindowSize = 30 # milliseconds
utteranceStepSize = utteranceWindowSize/2 # milliseconds
utteranceMinimumLength = 200 # milliseconds
utteranceMaximumVariance = 40
utteranceEnergyThreshold = 60

audioDirectory = "../media/Participant_Audio_First_five/*.wav"
outputDir = "./filledPauses/"

def createSlicesFromPauses():
    filledPausesAllParticipants = [["participant", "condition", "time", "judgement"]]

    # Has parameters and functions for voice activity in convenient package
    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    for filePath in sorted(glob.iglob(audioDirectory)):
        # Audio file i/o
        name = os.path.basename(filePath)[:-4]

        participant = name.split("_")[0]
        condition = name.split("_")[1]

        # # Make fresh directories
        # os.mkdir(outputDir + name)

        audio = audioModule.Audio(filePath=filePath)
        audio.makeMono()

        filledPauses, timeStamps, times, f1, f2, energy, lengths = featureModule.getFilledPauses(audio.data, audio.sampleRate, utteranceWindowSize, utteranceStepSize, utteranceMinimumLength, utteranceMaximumVariance, utteranceEnergyThreshold)

        audio = AudioSegment.from_wav(filePath)

        for time in timeStamps:
            filledPausesAllParticipants.append([participant, condition, str(round(time, 2)), ""])

            ### Output files — pydub is in ms
            outputPath = outputDir + name + "/" + str(round(time, 2))

            # move back 100 ms
            start = (time - 0.1) * 1000
            # grab a second
            end = (time + 1) * 1000
            segment = audio[start:end]

            # write to disk
            segment.export(outputPath + ".wav", format="wav")

            # move back a whole second for more context
            start = (time - 1) * 1000
            segment = audio[start:end]

            # write to disk
            segment.export(outputPath + "[extra].wav", format="wav")

            plt.figure()

            start = int(start/utteranceStepSize)
            end = int(end/utteranceStepSize)

            plt.plot(times[start:end], f1[start:end], times[start:end], f2[start:end], times[start:end], energy[start:end], time, 0, 'ro')

            plt.savefig(outputPath + "[extra].png")
            plt.close()
        # --

        print("Done with ", name)
        print(sum(filledPauses))
    # --

    # # Open File
    # outFile = open("filledPausesAllParticipants.csv",'w')
    #
    # # Write data to file
    # for row in filledPausesAllParticipants:
    #     outFile.write(", ".join(row) + "\n")
    #
    # outFile.close()
# --

def checkNewAlgorithmAgainstSlices():

    yeses = 0
    maybes = 0
    nos = 0

    for filePath in sorted(glob.iglob(audioDirectory)):
        # Audio file i/o
        name = os.path.basename(filePath)[:-4]

        print(name)

        participant = name.split("_")[0]
        condition = name.split("_")[1]

        audio = audioModule.Audio(filePath=filePath)
        audio.makeMono()

        # Run algorithm
        filledPauses, timeStamps, times, f1, f2, energy, lengths = featureModule.getFilledPauses(audio.data, audio.sampleRate, utteranceWindowSize, utteranceStepSize, utteranceMinimumLength, utteranceMaximumVariance, utteranceEnergyThreshold)

        # Compare with file of all existing
        with open('filledPausesAllParticipantsRatings.csv') as csvfile:
            reader = csv.DictReader(csvfile)

            # Go through each existing filled pause
            for row in reader:
                controlParticipant = row['participant']
                controlCondition = row['condition']
                controlTime = row['time']
                judgement = row['judgement']

                if controlParticipant == participant and controlCondition[1:] == condition:
                    name = participant + "_" + condition[1:]

                    found = False

                    for time in timeStamps:
                        if abs(time - float(controlTime)) < 0.01:
                            if judgement == "1":
                                yeses += 1
                            elif judgement == "0":
                                maybes += 1
                            elif judgement == "-1":
                                nos += 1

    # Print original accuracy
    with open('filledPausesAllParticipantsRatings.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        controlYeses = 0
        controlMaybes = 0
        controlNos = 0

        # Check accuracy of original
        for row in reader:
            judgement = row['judgement']

            if judgement == "1":
                controlYeses += 1
            elif judgement == "0":
                controlMaybes += 1
            elif judgement == "-1":
                controlNos += 1
        print()
        print("  Original yeses:", controlYeses, "maybes:", controlMaybes, "nos:", controlNos)

    print("  Yeses:", yeses, "maybes:", maybes, "nos:", nos)
    print()

def printParameters():
    print()
    print("  utteranceWindowSize:", utteranceWindowSize)
    print("  utteranceStepSize:", utteranceStepSize)
    print("  utteranceMinimumLength:", utteranceMinimumLength)
    print("  utteranceMaximumVariance:", utteranceMaximumVariance)
    print("  utteranceEnergyThreshold:", utteranceEnergyThreshold)
    print()

def main():
    # printParameters()
    # checkNewAlgorithmAgainstSlices()
    createSlicesFromPauses()

main()
