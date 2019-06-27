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

        times = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.voiceActivityStepSize / 1000)
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

        times = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.voiceActivityStepSize / 1000)
        signalTimes = np.arange(0, len(audio.data)/audio.sampleRate, 1 / audio.sampleRate)

        plt.figure(figsize=[16, 8])
        plt.plot(signalTimes, audio.data, times, voiceActivity)
        plt.title(name)
        plt.show()

def showSyllables():

    # filePath = "../media/cchp_english/p102/p102_en_pd.wav"
    filePath = "../media/Participant_Audio/p1_ol.wav"
    name = os.path.basename(filePath)[:-4]

    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    audio = audioModule.Audio(filePath=filePath)
    if audio.numberOfChannels != 1:
        audio.makeMono()

    audio.description()

    syllables, candidates = speechAnalyzer.getSyllablesWithPitchFromAudio(audio)
    print(len(syllables))
    syllableMarkers = np.full(len(syllables), 0)
    candidateMarkers = np.full(len(candidates), 0)

    voiceActivity = speechAnalyzer.getVoiceActivityFromAudio(audio)
    voiceActivityTimes = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.syllableStepSize/1000)[:len(pitch)]
    print(len(voiceActivity), len(voiceActivityTimes))

    ### Energy
    energy = librosa.feature.rmse(audio.data, frame_length=int(audio.sampleRate / 1000 * speechAnalyzer.syllableStepSize), hop_length=int(audio.sampleRate / 1000 * speechAnalyzer.syllableStepSize))[0]
    energyTimes = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.syllableStepSize/1000)[:len(energy)]

    pitch = featureModule.getPitchAC(audio.data, audio.sampleRate, speechAnalyzer.syllableStepSize)
    pitchTimes = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.syllableStepSize/1000)[:len(pitch)]

    voiceActivity = speechAnalyzer.getVoiceActivityFromAudio(audio)
    voiceActivity[voiceActivity == 0] = np.nan

    times = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.voiceActivityStepSize / 1000)
    signalTimes = np.arange(0, len(audio.data)/audio.sampleRate, 1 / audio.sampleRate)

    plt.figure(figsize=[16, 8])
    plt.plot(energyTimes, energy / 10, pitchTimes, pitch, candidates, candidateMarkers, 'ro')
    plt.plot(syllables, syllableMarkers, 'go')
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

        voiceActivity = speechAnalyzer.getVoiceActivityFromAudio(audio)
        voiceActivity[voiceActivity == 0] = np.nan

        syllables, candidates = speechAnalyzer.getSyllablesWithPitchFromAudio(audio)
        print(len(syllables))
        syllableMarkers = np.full(len(syllables), 0)
        candidateMarkers = np.full(len(candidates), 0)

        pitch = featureModule.getPitchAC(audio.data, audio.sampleRate, speechAnalyzer.syllableStepSize)
        pitchTimes = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.syllableStepSize/1000)[:len(pitch)]

        voiceActivity = speechAnalyzer.getVoiceActivityFromAudio(audio)
        print(np.sum(voiceActivity))
        voiceActivity[voiceActivity == 0] = np.nan

        times = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.voiceActivityStepSize / 1000)
        signalTimes = np.arange(0, len(audio.data)/audio.sampleRate, 1 / audio.sampleRate)

        plt.figure(figsize=[16, 8])
        plt.plot(signalTimes, audio.data, pitchTimes, pitch)
        plt.plot(syllables, syllableMarkers, 'go')
        plt.plot(times, voiceActivity)
        plt.title(name)
        plt.show()

def main():
    showVoiceActivityAndSyllablesForParticipantAudio()

main()
