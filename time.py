#
# Created on July 18, 2019
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

from speechLibrary import featureModule, speechAnalysis, audioModule

np.set_printoptions(threshold=sys.maxsize)

audioDirectory = "../media/validation_participant_audio/"

def intensity():
    times = []

    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    for filePath in sorted(glob.iglob(audioDirectory + "*.wav")):
        audio = audioModule.Audio(filePath=filePath)
        if audio.numberOfChannels != 1:
            audio.makeMono()

        startTime = time.time()
        _ = featureModule.getIntensityFeatures(audio.data)
        times.append(time.time() - startTime)

    print("  Intensity   | Time:", np.mean(times))

def pitch():
    times = []

    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    for filePath in sorted(glob.iglob(audioDirectory + "*.wav")):
        audio = audioModule.Audio(filePath=filePath)
        if audio.numberOfChannels != 1:
            audio.makeMono()

        startTime = time.time()
        _ = featureModule.getPitchFeatures(audio.data, audio.sampleRate, speechAnalyzer.voiceActivityStepSize)
        times.append(time.time() - startTime)

    print("    Pitch     | Time:", np.mean(times))

def voiceActivity():
    times = []

    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    for filePath in sorted(glob.iglob(audioDirectory + "*.wav")):
        audio = audioModule.Audio(filePath=filePath)
        if audio.numberOfChannels != 1:
            audio.makeMono()

        startTime = time.time()
        _ = speechAnalyzer.getVoiceActivityFromAudio(audio)
        times.append(time.time() - startTime)

    print("Voice activity| Time:", np.mean(times))


def syllables():
    times = []

    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    for filePath in sorted(glob.iglob(audioDirectory + "*.wav")):
        audio = audioModule.Audio(filePath=filePath)
        if audio.numberOfChannels != 1:
            audio.makeMono()

        startTime = time.time()
        _ = speechAnalyzer.getSyllablesFromAudio(audio)
        times.append(time.time() - startTime)

    print("  Syllables   | Time:", np.mean(times))

def allFeatures():
    times = []

    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    for filePath in sorted(glob.iglob(audioDirectory + "*.wav")):
        audio = audioModule.Audio(filePath=filePath)
        if audio.numberOfChannels != 1:
            audio.makeMono()

        startTime = time.time()
        _ = speechAnalyzer.getFeaturesFromAudio(audio)
        times.append(time.time() - startTime)

    print(" All Features | Time:", np.mean(times))

def main():
    # allFeatures()

    intensity()
    pitch()
    voiceActivity()
    syllables()

main()
