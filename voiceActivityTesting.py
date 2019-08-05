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
import random
import webrtcvad

from speechLibrary import featureModule, speechAnalysis, audioModule

sys.path.append("../validation/rVADfast_py_2.0/")
import rVAD_fast

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

def tunerVAD():
    thresholdValues = np.arange(1, 2.1, 0.1)
    print("threshold values:", thresholdValues)

    audioDirectory = "../media/validation_participant_audio/"
    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    results = []

    for threshold in thresholdValues:
        print(threshold)

        transcript = []

        totalNumberOfVoiceActivityInstances = 0
        totalNumberOfCorrectlyDetectedVoiceActivityInstances = 0
        totalNumberOfFalseAlarms = 0

        with open(audioDirectory + "/voice_activity.txt") as transcriptFile:
            lines = transcriptFile.readlines()
            for row in lines:
                transcript.append(row.strip().split(', '))

        # linesToCheck = list(range(0,len(transcript)))
        # random.shuffle(linesToCheck)
        #
        # print(linesToCheck[:10])
        #
        # for lineIndex in linesToCheck[:20]:
        #     line = transcript[lineIndex]
        for line in transcript:
            name = line[0]

            if name[0] != "#":
                actualVoiceActivity = [int(element) for element in line[1:31]]

                # Find the match audio file
                for filePath in sorted(glob.iglob(audioDirectory + "*.wav")):
                    fileName = os.path.basename(filePath)[:-4]

                    if fileName == name:
                        print(fileName)

                        audio = audioModule.Audio(filePath=filePath)
                        if audio.numberOfChannels != 1:
                            audio.makeMono()

                        rVADVoiceActivity = rVAD_fast.voiceActivity(finwav= filePath,
                                                                    vadThres=threshold)

                        rVADVoiceActivityBins = []

                        frameSizeInSeconds = 1
                        frameSizeInSteps = int(frameSizeInSeconds / (speechAnalyzer.featureStepSize / 1000))
                        voiceActivity = []

                        originalNumberOfFalseAlarms = totalNumberOfFalseAlarms

                        for frame in range (0, int(len(audio.data) / audio.sampleRate / (frameSizeInSeconds))):
                            start = frame * frameSizeInSteps
                            end = frame * frameSizeInSteps + frameSizeInSteps

                            rVADVoiceActivityValue = int(max(rVADVoiceActivity[start:end]))
                            rVADVoiceActivityBins.append(rVADVoiceActivityValue)

                            if rVADVoiceActivityValue == 1:
                                if actualVoiceActivity[frame] == 0:
                                    totalNumberOfFalseAlarms += 1
                                if actualVoiceActivity[frame] == 1:
                                    totalNumberOfCorrectlyDetectedVoiceActivityInstances += 1

                        print("    actual ", end="")
                        for element in actualVoiceActivity:
                            if element == 0:
                                print("-", end="")
                            else:
                                print("█", end="")
                        print()
                        print("      rVAD ", end="")
                        for element in rVADVoiceActivityBins:
                            if element == 0:
                                print("-", end="")
                            else:
                                print("█", end="")
                        print()

                        totalNumberOfVoiceActivityInstances += int(sum(actualVoiceActivity))

        precision = totalNumberOfCorrectlyDetectedVoiceActivityInstances / (totalNumberOfCorrectlyDetectedVoiceActivityInstances + totalNumberOfFalseAlarms)
        recall = totalNumberOfCorrectlyDetectedVoiceActivityInstances / totalNumberOfVoiceActivityInstances

        fMeasure = 2 * precision * recall / (precision + recall)

        print("   Actual     | Seconds with voice activity:", totalNumberOfVoiceActivityInstances)
        print("  Algorithm   | Correct detectsions:", totalNumberOfCorrectlyDetectedVoiceActivityInstances, "False alarms:", totalNumberOfFalseAlarms, "Precision:", precision, "Recall:", recall, "F-measure", fMeasure)

        results.append([threshold, fMeasure])


def testWebRTCVAD(wavFile):
    sampleRateForDownsampling = [8000, 16000, 32000, 48000][2]
    windowSize = 10 # MS

    vad = webrtcvad.Vad()

    audio = audioModule.Audio(filePath= wavFile)
    audio.makeMono()

    print(audio.data.shape)

    audio.data = librosa.core.resample(audio.data, audio.sampleRate, sampleRateForDownsampling)
    audio.sampleRate = sampleRateForDownsampling

    print(audio.data.shape)

    windowSizeInSamples = int(audio.sampleRate / 1000 * windowSize)

    # # Pad the end of the array
    # audio.data = np.pad(data, (0, int(windowSizeInSamples - stepSizeInSamples)), mode='constant')

    # Create frames using optimized algorithm
    framedData = librosa.util.frame(audio.data,
                                    frame_length=windowSizeInSamples,
                                    hop_length=windowSizeInSamples)

    print(framedData.shape)

    voiceActivity = []

    for frame in framedData:
        print(frame.shape)
        frameData = frame.copy(order='C')
        binData = wavio._array2wav(frame, 2)
        print(binData)
        voiceActivity = [1 if vad.is_speech(f, vad.is_speech(binData, audio.sampleRate)) else 0 for f in frames]

    return np.array(voiceActivity)

def main():
    tunerVAD()

main()
