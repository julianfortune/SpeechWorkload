import parselmouth
import numpy as np
import matplotlib.pyplot as plt

import sys, time, glob, os

import wavio
import math
import librosa
import csv

from pydub import AudioSegment
import logging

np.set_printoptions(threshold=sys.maxsize)

# Returns the first two formant from a piece of audio.
def getFormants(data, sampleRate, windowSize, stepSize):
    # Convert to the parselmouth custom sound type (req'd for formant function)
    parselSound = parselmouth.Sound(values=data, sampling_frequency=sampleRate)

    # Produce a formant object from this data
    formantData = parselSound.to_formant_burg(window_length=windowSize, time_step=stepSize)

    # API for parselmouth doesn't explain how to extract the formatData without querying for everything manually.
    firstFormant = []
    secondFormant = []

    # Used for plots
    times = np.arange(0, len(data)/sampleRate, stepSize) # seconds

    for timeStamp in times:
        firstFormantValue = formantData.get_value_at_time(1, timeStamp)
        secondFormantValue = formantData.get_value_at_time(2, timeStamp)

        firstFormant.append(firstFormantValue)
        secondFormant.append(secondFormantValue)

    firstFormant = np.array(firstFormant)
    secondFormant = np.array(secondFormant)

    return firstFormant, secondFormant

# Returns a numpy array
def getFilledPauses(data, sampleRate, windowSize, stepSize, minumumLength, maximumVariance, energyThreshold):
    # Everything needs to be based on samples to prevent rounding issues with RMSE
    sampleWindowSize = int(windowSize*sampleRate/1000)
    sampleStepSize = int(stepSize*sampleRate/1000)

    # The number of steps in energy and formant arrays
    numberOfSteps = round((len(data)/sampleRate) / (sampleStepSize/sampleRate))

    # Formant extraction
    firstFormant, secondFormant = getFormants(data, sampleRate, sampleWindowSize/sampleRate, sampleStepSize/sampleRate)

    # Energy
    energy = librosa.feature.rmse(data, frame_length=sampleWindowSize, hop_length=sampleStepSize)[0]

    # Filled pauses detection
    filledPauses = np.zeros(numberOfSteps)
    timeStamps = []

    # Used for plots
    times = np.arange(0, len(data)/sampleRate, sampleStepSize/sampleRate) # seconds

    # # Needed for pretty graphs
    # filledPauses.fill(np.nan)

    # The number of steps in the feature arrays that make up a single window for checking for utterances.
    utteranceWindowSize = int(minumumLength / 1000 * 44100 / sampleStepSize)

    fillerUtteranceInitiated = False

    # Step through each data point in formant and energy arrays and check for
    # filler utterances over the next 'minimumLength' size window of features.
    for step in range(0, numberOfSteps-utteranceWindowSize):
        start = step
        end = step+utteranceWindowSize

        firstFormantVariance = np.std(firstFormant[start:end])
        secondFormantVariance = np.std(secondFormant[start:end])
        averageEnergy = np.mean(energy[start:end])

        if firstFormantVariance <= maximumVariance and secondFormantVariance <= maximumVariance and averageEnergy > energyThreshold:
            # Prevent an utterance from being detected many times
            if fillerUtteranceInitiated == False:
                filledPauses[step] = 1
                fillerUtteranceInitiated = True
                timeStamps.append(times[step])
        else:
            fillerUtteranceInitiated = False

    # # Used for plots
    # times = np.arange(0, len(data)/sampleRate, sampleStepSize/sampleRate) # seconds
    # # Graph it!
    # plt.plot(times, firstFormant, times, secondFormant, times, energy, times, filledPauses, 'ro')
    # # plt.title(file)
    # plt.show()

    return filledPauses, np.array(timeStamps)

def main():
    # Parameters of the features
    utteranceWindowSize = 30 # milliseconds
    utteranceStepSize = utteranceWindowSize/2 # milliseconds
    utteranceMinimumLength = 200 # milliseconds
    utteranceMaximumVariance = 40
    utteranceEnergyThreshold = 60

    audioDirectory = "../media/Participant_Audio/*.wav"
    outputDir = "./filledPauses/"

    filledPausesAllParticipants = [["participant", "condition", "time", "judgement"]]

    for filePath in sorted(glob.iglob(audioDirectory)):
        # Audio file i/o
        name = os.path.basename(filePath)[:-4]

        audio = wavio.read(filePath) # reads in audio file
        sampleRate = audio.rate
        data = np.mean(audio.data,axis=1)

        filledPauses, times = getFilledPauses(data, sampleRate, utteranceWindowSize, utteranceStepSize, utteranceMinimumLength, utteranceMaximumVariance, utteranceEnergyThreshold)

        audio = AudioSegment.from_wav(filePath)

        for time in times:
            participant = name.split("_")[0]
            condition = name.split("_")[1]

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
        # --

        print("Done with ", name)
    # --

    # Open File
    outFile = open("filledPausesAllParticipants.csv",'w')

    # Write data to file
    for row in filledPausesAllParticipants:
        outFile.write(", ".join(row) + "\n")

    outFile.close()
# --

main()
