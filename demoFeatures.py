#
# Created on June 25, 2019
#
# @author: Julian Fortune
# @Description: Functions for visualizing and analyzing features run on
# different audio recordings/sets of audio recordings.
#

import sys, time, glob, os, csv
import numpy as np
import matplotlib.pyplot as plt

import librosa

from speechLibrary import featureModule, speechAnalysis, audioModule

np.set_printoptions(threshold=sys.maxsize)

def showVoiceActivityForPNNC():

    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    audioDirectory = "../media/pnnc-v1/audio/*.wav"

    for filePath in sorted(glob.iglob(audioDirectory)):
        participant = filePath.split("/")[4][:5]
        name = filePath.split("/")[4]

        audio = audioModule.Audio(filePath=filePath)
        if audio.numberOfChannels != 1:
            audio.makeMono()

        voiceActivity = speechAnalyzer.getVoiceActivityFromAudio(audio)
        voiceActivity[voiceActivity == 0] = np.nan

        times = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.featureStepSize / 1000)
        signalTimes = np.arange(0, len(audio.data)/audio.sampleRate, 1 / audio.sampleRate)

        plt.figure(figsize=[16, 8])
        plt.plot(signalTimes, audio.data, times, voiceActivity)
        plt.title(name)
        plt.show()

def showVoiceActivityForParticipantAudio():

    audioDirectory = "../media/Participant_Audio/*.wav"

    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    for filePath in sorted(glob.iglob(audioDirectory)):
        name = os.path.basename(filePath)[:-4]

        audio = audioModule.Audio(filePath=filePath)
        if audio.numberOfChannels != 1:
            audio.makeMono()

        voiceActivity = speechAnalyzer.getVoiceActivityFromAudio(audio)
        voiceActivity[voiceActivity == 0] = np.nan

        times = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.featureStepSize / 1000)
        signalTimes = np.arange(0, len(audio.data)/audio.sampleRate, 1 / audio.sampleRate)

        plt.figure(figsize=[16, 8])
        plt.plot(signalTimes, audio.data, times, voiceActivity)
        plt.title(name)
        plt.show()

def showSyllables():

    # filePath = "../media/cchp_english/p102/p102_en_pd.wav"
    filePath = "../media/Participant_Audio_30_Sec_Chunks/p14_ol_chunk18.wav"
    name = os.path.basename(filePath)[:-4]

    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    audio = audioModule.Audio(filePath=filePath)
    if audio.numberOfChannels != 1:
        audio.makeMono()

    audio.description()

    syllables, candidates = speechAnalyzer.getSyllablesFromAudio(audio)
    print(len(syllables))
    syllableMarkers = np.full(len(syllables), 0)
    candidateMarkers = np.full(len(candidates), 0)

    ### Energy
    energy = librosa.feature.rmse(audio.data, frame_length=int(audio.sampleRate / 1000 * speechAnalyzer.featureStepSize), hop_length=int(audio.sampleRate / 1000 * speechAnalyzer.featureStepSize))[0]
    energyTimes = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.featureStepSize/1000)[:len(energy)]

    energyMinThreshold = featureModule.getEnergyMinimumThreshold(energy)
    fractionEnergyMinThreshold = energyMinThreshold / max(energy)

    pitch = featureModule.getPitch(audio.data, audio.sampleRate, speechAnalyzer.featureStepSize, fractionEnergyMinThreshold)
    pitchTimes = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.featureStepSize/1000)[:len(pitch)]

    zcr = librosa.feature.zero_crossing_rate(audio.data, frame_length=int(audio.sampleRate / 1000 * speechAnalyzer.featureStepSize * 4), hop_length=int(audio.sampleRate / 1000 * speechAnalyzer.featureStepSize))[0]
    zcrTimes = np.arange(0, len(audio.data)/audio.sampleRate + 1, speechAnalyzer.featureStepSize/1000)[:len(zcr)]

    voiceActivity = speechAnalyzer.getVoiceActivityFromAudio(audio)
    voiceActivity[voiceActivity == 0] = np.nan
    voiceActivityTimes = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.featureStepSize/1000)[:len(voiceActivity)]
    print(len(voiceActivity), len(voiceActivityTimes))

    times = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.featureStepSize / 1000)
    signalTimes = np.arange(0, len(audio.data)/audio.sampleRate, 1 / audio.sampleRate)

    plt.figure(figsize=[16, 8])
    plt.plot(energyTimes, energy / 10, pitchTimes, pitch, zcrTimes, zcr * 100, candidates, candidateMarkers, 'ro')
    plt.plot(syllables, syllableMarkers, 'go')
    plt.plot(voiceActivityTimes, voiceActivity)
    plt.title(name)
    plt.show()

def showVoiceActivityAndSyllablesForParticipantAudio():
    audioDirectory = "../media/Participant_Audio/*.wav"

    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    for filePath in sorted(glob.iglob(audioDirectory)):
        name = os.path.basename(filePath)[:-4]

        audio = audioModule.Audio(filePath=filePath)
        if audio.numberOfChannels != 1:
            audio.makeMono()

        print("Getting voice activity...")

        voiceActivity = speechAnalyzer.getVoiceActivityFromAudio(audio)
        voiceActivity[voiceActivity == 0] = np.nan

        voiceActivityBufferSize = int(100 / speechAnalyzer.featureStepSize)
        voiceActivityBuffered = featureModule.createBufferedBinaryArrayFromArray(voiceActivity == 1, voiceActivityBufferSize).astype(int).astype(float)
        voiceActivityBuffered[voiceActivityBuffered == 0] = np.nan

        print("Getting syllables...")

        syllables, _ = speechAnalyzer.getSyllablesFromAudio(audio)
        syllableMarkers = np.full(len(syllables), 0)

        print("Getting other features...")

        energy = featureModule.getEnergy(audio.data, audio.sampleRate, speechAnalyzer.syllableWindowSize, speechAnalyzer.featureStepSize)
        energyMinThreshold = featureModule.getEnergyMinimumThreshold(energy)
        fractionEnergyMinThreshold = energyMinThreshold / max(energy)

        zcr = librosa.feature.zero_crossing_rate(audio.data, frame_length=int(audio.sampleRate / 1000 * speechAnalyzer.featureStepSize), hop_length=int(audio.sampleRate / 1000 * speechAnalyzer.featureStepSize))[0]
        zcrTimes = np.arange(0, len(audio.data)/audio.sampleRate + 1, speechAnalyzer.featureStepSize/1000)[:len(zcr)]

        pitch = speechAnalyzer.getPitchFromAudio(audio)
        pitch[pitch==0] = np.nan
        pitchTimes = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.featureStepSize/1000)[:len(pitch)]

        times = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.featureStepSize / 1000)
        energyTimes = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.featureStepSize / 1000)[:len(energy)]

        print("Graphing!")

        plt.figure(figsize=[16, 8])
        plt.plot(zcrTimes, zcr * 1000, 'gold')
        plt.plot(times, energy / 5)
        plt.plot(pitchTimes, pitch, 'red')
        plt.plot(syllables, syllableMarkers, 'go')
        plt.plot(times, voiceActivityBuffered * -5, 'darkorange')
        plt.plot(times, voiceActivity, 'purple')
        plt.title(name)
        plt.show()

def getFeaturesFromFileUsingWindowing():
    filePath = "../media/Participant_Audio/p3_ol.wav"
    name = os.path.basename(filePath)[:-4]

    speechAnalyzer = speechAnalysis.SpeechAnalyzer()
    speechAnalyzer.lookBackSize = 5

    # Read in the file, extract data and metadata
    audio = audioModule.Audio(filePath)
    if audio.numberOfChannels > 1:
        audio.makeMono()

    # Set up time tracker
    seconds = np.zeros(shape=0)

    step = 0
    sampleStepSize = int(speechAnalyzer.stepSize * audio.sampleRate)
    sampleLookBackSize = int(speechAnalyzer.lookBackSize * audio.sampleRate)

    while step < audio.length:
        # Keep track of what second we're in
        print("Second:", step/audio.sampleRate)

        # Look backward to calculate features over long term
        if step + sampleStepSize - sampleLookBackSize > 0:

            currentWindow = audioModule.Audio(data=audio.data[step + sampleStepSize - sampleLookBackSize:step + sampleStepSize])
            currentWindow.sampleRate = audio.sampleRate

            ### WORDS PER MINUTE
            syllables = speechAnalyzer.getSyllablesFromAudio(currentWindow)[0]
            syllableMarkers = np.full(len(syllables), 0)

            ### VAD
            voiceActivity = speechAnalyzer.getVoiceActivityFromAudio(currentWindow)

            ### INTENSITY
            energy = featureModule.getEnergy(currentWindow.data, currentWindow.sampleRate, speechAnalyzer.syllableWindowSize, speechAnalyzer.featureStepSize)

            energyMinThreshold = featureModule.getEnergyMinimumThreshold(energy)
            fractionEnergyMinThreshold = energyMinThreshold / max(energy)

            ### PITCH
            pitch = featureModule.getPitch(currentWindow.data, currentWindow.sampleRate, speechAnalyzer.featureStepSize, fractionEnergyMinThreshold)

            syllableBinaryArray = np.full(len(voiceActivity), 0)

            for timeStamp in syllables:
                syllableBinaryArray[int(timeStamp / (currentWindow.sampleRate / 1000 * speechAnalyzer.featureStepSize) * currentWindow.sampleRate)] = 1

            # Mask out all filled pauses that coincide with voice acitivty
            syllableBinaryArray[voiceActivity.astype(bool)] = 0

            if max(syllableBinaryArray) >= 1:

                # Clean up va for graphing
                voiceActivity[voiceActivity == 0] = np.nan
                pitch[pitch == 0] = np.nan

                pitchTimes = np.arange(0, len(currentWindow.data)/currentWindow.sampleRate, speechAnalyzer.featureStepSize/1000)[:len(pitch)]
                energyTimes = np.arange(0, len(currentWindow.data)/currentWindow.sampleRate, speechAnalyzer.featureStepSize/1000)[:len(energy)]
                times = np.arange(0, len(currentWindow.data)/currentWindow.sampleRate, speechAnalyzer.featureStepSize / 1000)

                plt.figure(figsize=[16, 8])
                plt.plot(times, energy / 10, pitchTimes, pitch)
                plt.plot(times, voiceActivity)
                plt.plot(syllables, syllableMarkers, 'r^')
                plt.title(name + " from " + str(step/audio.sampleRate - speechAnalyzer.lookBackSize) + " to " + str(step/audio.sampleRate) + " seconds")
                # plt.savefig("./syllablesVersusVAD/" + name + "_" + str(step/audio.sampleRate - speechAnalyzer.lookBackSize) + "-" + str(step/audio.sampleRate) + "_seconds.png")
                plt.show()


        # Increment to next step
        step += sampleStepSize

def main():
    showVoiceActivityAndSyllablesForParticipantAudio()

main()
