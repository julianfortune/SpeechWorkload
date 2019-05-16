import parselmouth
import numpy as np
import matplotlib.pyplot as plt

import sys, time

import wavio
import math
import librosa

np.set_printoptions(threshold=sys.maxsize)

start = time.time()

# Parameters of the features

utteranceWindowSize = 30 # milliseconds
utteranceStepSize = utteranceWindowSize/2 # milliseconds

utteranceMinimumLength = 200 # milliseconds
utteranceMaximumVariance = 40
utteranceEnergyThreshold = 60


# Audio file i/o

file = "p1_ol.wav"

filePath = "../media/Participant_Audio/" + file

inputSound = parselmouth.Sound(filePath)

audio = wavio.read(filePath) # reads in audio file
sampleRate = audio.rate
data = np.mean(audio.data,axis=1)

print("Loaded file")
print(time.time() - start)

def getFilledPauses(data, windowSize, stepSize, minumumLength, maximumVariance, energyThreshold):

    # Helper parameter set up

    # Everything needs to be based on samples to prevent rounding issues with RMSE
    sampleWindowSize = int(windowSize*sampleRate/1000)
    sampleStepSize = int(stepSize*sampleRate/1000)

    times = np.arange(0, len(data)/sampleRate, sampleStepSize/sampleRate) # seconds

    print("Getting features")
    print(time.time() - start)

    # --- Core feature extraction ---

    # Formant extraction

    formantData = inputSound.to_formant_burg(window_length=sampleWindowSize/sampleRate, time_step=sampleStepSize/sampleRate)

    firstFormant = []
    secondFormant = []

    for timeStamp in times:
        firstFormantValue = formantData.get_value_at_time(1, timeStamp)
        secondFormantValue = formantData.get_value_at_time(2, timeStamp)

        firstFormant.append(firstFormantValue)
        secondFormant.append(secondFormantValue)

    firstFormant = np.array(firstFormant)
    secondFormant = np.array(secondFormant)

    # Energy

    energy = librosa.feature.rmse(data, frame_length=sampleWindowSize, hop_length=sampleStepSize)[0][:len(times)]

    print("Getting filled pauses")
    print(time.time() - start)

    # Filled pauses detection

    filledPauses = np.empty(len(times))
    filledPauses.fill(np.nan)

    frameWindowSize = int(minumumLength / 1000 * 44100 / sampleStepSize)

    fillerUtteranceInitiated = False

    for frame in range(0, len(times)-frameWindowSize):
        firstVariance = np.std(firstFormant[frame:frame+frameWindowSize])
        secondVariance = np.std(secondFormant[frame:frame+frameWindowSize])
        averageEnergy = np.mean(energy[frame:frame+frameWindowSize])

        if firstVariance <= maximumVariance and secondVariance <= maximumVariance and averageEnergy > energyThreshold:
            if fillerUtteranceInitiated == False:
                filledPauses[frame] = 1
                fillerUtteranceInitiated = True
        else:
            fillerUtteranceInitiated = False

    plt.plot(times, firstFormant, times, secondFormant, times, energy, times, filledPauses, 'ro')
    plt.title(file)
    plt.show()


getFilledPauses(data, utteranceWindowSize, utteranceStepSize, utteranceMinimumLength, utteranceMaximumVariance, utteranceEnergyThreshold)
