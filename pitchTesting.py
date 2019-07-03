import sys, time, glob, os
import numpy as np
import matplotlib.pyplot as plt

import librosa

from speechLibrary import featureModule, speechAnalysis, audioModule

np.set_printoptions(threshold=sys.maxsize)

def compareAveragePitchOnPNNC():

    audioDirectory = "../media/pnnc-v1/audio/*.wav"

    stepSize = 20
    participantPitches = {}
    graph = True

    for filePath in sorted(glob.iglob(audioDirectory)):
        participant = filePath.split("/")[4][:5]
        name = filePath.split("/")[4]

        audio = audioModule.Audio(filePath=filePath)
        if audio.numberOfChannels != 1:
            audio.makeMono()

        pitch = featureModule.getPitch(data=audio.data, sampleRate=audio.sampleRate, stepSize=stepSize)

        zcr = librosa.feature.zero_crossing_rate(audio.data, frame_length=int(audio.sampleRate / 1000 * 10), hop_length=int(audio.sampleRate / 1000 * stepSize) )[0]

        energy = librosa.feature.rmse(audio.data, frame_length=int(audio.sampleRate / 1000 * 50), hop_length=int(audio.sampleRate / 1000 * stepSize) )[0]

        if graph:
            pitchTimes = np.arange(0, len(audio.data)/audio.sampleRate, stepSize/1000)[:len(pitch)]
            zcrTimes = np.arange(0, len(audio.data)/audio.sampleRate, stepSize/1000)[:len(zcr)]
            plt.figure(figsize=[16, 8])
            plt.plot(pitchTimes, pitch, zcrTimes, zcr * 500, zcrTimes, energy / 10)
            plt.title(name)
            plt.show()

        if participant in participantPitches:
            participantPitches[participant].append(np.nanmean(pitch))
        else:
            participantPitches[participant] = [np.nanmean(pitch)]
            print(participant)

    for participantName in participantPitches:
        print(participantName, np.mean(participantPitches[participantName]), np.stdev(participantPitches[participantName]))

def compareAveragePitchOnParticipantAudio():

    audioDirectory = "../media/Participant_Audio/*.wav"

    stepSize = 10
    graph = False

    for filePath in sorted(glob.iglob(audioDirectory)):
        name = os.path.basename(filePath)[:-4]

        audio = audioModule.Audio(filePath=filePath)
        if audio.numberOfChannels != 1:
            audio.makeMono()

        pitch = featureModule.getPitch(data=audio.data, sampleRate=audio.sampleRate, stepSize=stepSize)

        if graph:
            times = np.arange(0, len(audio.data)/audio.sampleRate, stepSize/1000)[:len(pitch)]
            plt.plot(times, pitch)
            plt.show()

        onlyVoicedPitches = pitch[np.isfinite(pitch)]

        if len(onlyVoicedPitches) > 0:
            print(name, np.mean(onlyVoicedPitches), np.std(onlyVoicedPitches), max(onlyVoicedPitches), min(onlyVoicedPitches))
        else:
            print(name)

def main():
    compareAveragePitchOnPNNC()

main()
