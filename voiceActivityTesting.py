#
# Created on July 8, 2019
#
# @author: Julian Fortune
# @Description: Functions for testing and validating the voice activity algorithm.
#

import sys, time, glob, os
import numpy as np
import matplotlib.pyplot as plt

import librosa

from speechLibrary import featureModule, speechAnalysis, audioModule

np.set_printoptions(threshold=sys.maxsize)

def compareVoiceActivityToParticipants():
    audioDirectory = "../media/Participant_Audio_30_Sec_Chunks/*.wav"
    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    transcript = []

    totalNumberOfVoiceActivityInstances = 0
    totalNumberOfCorrectlyDetectedVoiceActivityInstances = 0
    totalNumberOfFalseAlarms = 0

    with open("../media/Participant_Audio_30_Sec_Chunks_Transcripts/voice_activity.txt") as transcriptFile:
        lines = transcriptFile.readlines()
        for row in lines:
            transcript.append(row.strip().split(', '))

    for line in transcript:
        name = line[0]

        if name[0] != "#":
            actualVoiceActivity = [int(element) for element in line[1:31]]

            for filePath in sorted(glob.iglob(audioDirectory)):
                fileName = os.path.basename(filePath)[:-4]

                if fileName == name:
                    audio = audioModule.Audio(filePath=filePath)
                    if audio.numberOfChannels != 1:
                        audio.makeMono()

                    rawVoiceActivity = speechAnalyzer.getVoiceActivityFromAudio(audio)

                    # voiceActivityBufferSize = int(100 / speechAnalyzer.voiceActivityStepSize)
                    # voiceActivityBuffered = featureModule.createBufferedBinaryArrayFromArray(voiceActivity == 1, voiceActivityBufferSize)

                    frameSizeInSeconds = 1
                    frameSizeInSteps = int(frameSizeInSeconds / (speechAnalyzer.featureStepSize / 1000))
                    voiceActivity = []

                    for frame in range (0, int(len(audio.data) / audio.sampleRate / (frameSizeInSeconds))):
                        voiceActivityValue = int(max(rawVoiceActivity[frame * frameSizeInSteps:frame * frameSizeInSteps + frameSizeInSteps]))
                        voiceActivity.append(voiceActivityValue)

                        if voiceActivityValue == 1:
                            if actualVoiceActivity[frame] == 0:
                                totalNumberOfFalseAlarms += 1
                            if actualVoiceActivity[frame] == 1:
                                totalNumberOfCorrectlyDetectedVoiceActivityInstances += 1

                    if voiceActivity != actualVoiceActivity:
                        print(name, "x")
                        print(actualVoiceActivity)
                        print(voiceActivity)
                    else:
                        print(name, "✓")
                    print(sum(rawVoiceActivity))

                    totalNumberOfVoiceActivityInstances += int(sum(actualVoiceActivity))


    precision = totalNumberOfCorrectlyDetectedVoiceActivityInstances / (totalNumberOfCorrectlyDetectedVoiceActivityInstances + totalNumberOfFalseAlarms)
    recall = totalNumberOfCorrectlyDetectedVoiceActivityInstances / totalNumberOfVoiceActivityInstances

    fMeasure = 2 * precision * recall / (precision + recall)

    print("   Actual     | Seconds with voice activity:", totalNumberOfVoiceActivityInstances)
    print("  Algorithm   | Correct detectsions:", totalNumberOfCorrectlyDetectedVoiceActivityInstances, "Precision:", precision, "Recall:", recall, "F-measure", fMeasure)

def voiceActivityRuns():
    filePath = "../media/Participant_Audio/p10_ol.wav"
    name = os.path.basename(filePath)[:-4]

    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    audio = audioModule.Audio(filePath=filePath)
    if audio.numberOfChannels != 1:
        audio.makeMono()

    voiceActivity = speechAnalyzer.getVoiceActivityFromAudio(audio)
    print("Total voice activity:", sum(voiceActivity))

    runs = 0
    voiceActivityPresent = False
    for frame in voiceActivity:
        if not voiceActivityPresent and frame == 1:
            voiceActivityPresent = True
            runs += 1
        if voiceActivityPresent and frame == 0:
            voiceActivityPresent = False
    print("Total number of segments of voice activity:", runs)

def main():
    compareVoiceActivityToParticipants()

main()
