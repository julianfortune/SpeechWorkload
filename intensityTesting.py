#
# Created on July 5, 2019
#
# @author: Julian Fortune
# @Description: Functions for testing different intensity/power/energy
# algorithms.
#

import sys, time, glob, os
import numpy as np
import matplotlib.pyplot as plt

import librosa
import parselmouth
import math

from speechLibrary import featureModule, speechAnalysis, audioModule

np.set_printoptions(threshold=sys.maxsize)

def compareEnergyAndIntensity():
    filePath = "../media/Participant_Audio/p10_ol.wav"
    name = os.path.basename(filePath)[:-4]

    stepSize = 10 # In milliseconds
    windowSize = 10

    audio = audioModule.Audio(filePath=filePath)
    if audio.numberOfChannels != 1:
        audio.makeMono()

    stepSizeInSamples = int(audio.sampleRate / 1000 * stepSize)
    windowSizeInSamples = int(audio.sampleRate / 1000 * windowSize)

    # Parselmouth intensity
    parselSound = parselmouth.Sound(values=audio.data, sampling_frequency=audio.sampleRate)
    intensityObject = parselSound.to_intensity(minimum_pitch= 50.0, time_step=stepSize/1000)
    intensity = intensityObject.values.T

    shortTermEnergy = np.array([
        math.sqrt(sum(audio.data[step:step+windowSizeInSamples]**2) / windowSizeInSamples)
        for step in range(0, len(audio.data), stepSizeInSamples)
    ])

    rms = np.array([
        sum(audio.data[step:step+windowSizeInSamples]**2)
        for step in range(0, len(audio.data), stepSizeInSamples)
    ])

    # Librosa rms
    rms = librosa.feature.rms(audio.data, frame_length=windowSizeInSamples, hop_length=stepSizeInSamples)[0]

    # Current intensity measure
    amplitude = np.absolute(audio.data)

    intensityTimes = np.arange(0, len(audio.data)/audio.sampleRate, stepSize/1000)[:len(intensity)]
    shortTermEnergyTimes = np.arange(0, len(audio.data)/audio.sampleRate, stepSize/1000)[:len(shortTermEnergy)]
    rmsTimes = np.arange(0, len(audio.data)/audio.sampleRate, stepSize/1000)[:len(rms)]
    signalTimes = np.arange(0, len(audio.data)/audio.sampleRate, 1 / audio.sampleRate)

    plt.figure(figsize=[16, 8])
    # plt.plot(signalTimes, amplitude / 2)
    plt.plot(shortTermEnergyTimes, shortTermEnergy)
    plt.plot(rmsTimes, rms)
    plt.plot(intensityTimes, intensity * 100)
    plt.title(name)
    plt.show()

def compareLibrosaAndRMS():
    filePath = "../media/Participant_Audio/p10_ol.wav"
    name = os.path.basename(filePath)[:-4]

    stepSize = 10 # In milliseconds
    windowSize = 10

    audio = audioModule.Audio(filePath=filePath)
    if audio.numberOfChannels != 1:
        audio.makeMono()

    librosaRMS = featureModule.getEnergy(data= audio.data,
                                         sampleRate= audio.sampleRate,
                                         windowSize= windowSize,
                                         stepSize= stepSize)

    rms = featureModule.getRMSIntensity(data= audio.data,
                                        sampleRate= audio.sampleRate,
                                        windowSize= windowSize,
                                        stepSize= stepSize)

    times = np.arange(0, len(audio.data)/audio.sampleRate, stepSize/1000)

    plt.figure(figsize=[16, 8])
    plt.plot(times, librosaRMS)
    plt.plot(times, rms)
    plt.title(name)
    plt.show()


def main():
    compareLibrosaAndRMS()

main()
