import numpy as np
import matplotlib.pyplot as plt

import sys, time, glob, os

import wavio
import csv

from pydub import AudioSegment

from speechLibrary import featureModule, speechAnalysis, audioModule

np.set_printoptions(threshold=sys.maxsize)

def main():
    # Parameters of the features
    utteranceWindowSize = 30 # milliseconds
    utteranceStepSize = utteranceWindowSize/2 # milliseconds
    utteranceMinimumLength = 200 # milliseconds
    utteranceMaximumVariance = 40
    utteranceEnergyThreshold = 60

    audioDirectory = "../media/Participant_Audio_First_five/*.wav"
    outputDir = "./filledPauses/"

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

main()
