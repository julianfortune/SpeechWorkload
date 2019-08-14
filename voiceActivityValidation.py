#
# Created on July 30, 2019
#
# @author: Julian Fortune
# @Description:
#

import sys, time, glob, os
import numpy as np
import matplotlib.pyplot as plt

from speechLibrary import featureModule, speechAnalysis, audioModule

sys.path.append("../validation/rVADfast_py_2.0/")
import rVAD_fast

np.set_printoptions(threshold=sys.maxsize)

def validateOnCherrypickedSet():
    audioDirectory = "../media/validation_participant_audio/"

    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    transcript = []

    totalNumberOfVoiceActivityInstances = 0
    totalNumberOfCorrectlyDetectedVoiceActivityInstances = 0
    totalNumberOfFalseAlarms = 0

    with open(audioDirectory + "/voice_activity.txt") as transcriptFile:
        lines = transcriptFile.readlines()
        for row in lines:
            transcript.append(row.strip().split(', '))

    times = []

    for line in transcript:
        name = line[0]

        # if name == "p30_ul_20.01-50.01" or name == "p30_ol_47.31-77.31":
        if name[0] != "#":
            actualVoiceActivity = [int(element) for element in line[1:31]]

            # Find the match audio file
            for filePath in sorted(glob.iglob(audioDirectory + "*.wav")):
                fileName = os.path.basename(filePath)[:-4]

                if fileName == name:
                    # Import file
                    audio = audioModule.Audio(filePath=filePath)
                    if audio.numberOfChannels != 1:
                        audio.makeMono()

                    pitches = speechAnalyzer.getPitchFromAudio(audio)

                    startTime = time.time()
                    rawVoiceActivity = speechAnalyzer.getVoiceActivityFromAudio(audio, pitches= pitches)
                    times.append(time.time() - startTime)

                    frameSizeInSeconds = 1
                    frameSizeInSteps = int(frameSizeInSeconds / (speechAnalyzer.featureStepSize / 1000))
                    voiceActivity = []

                    originalNumberOfFalseAlarms = totalNumberOfFalseAlarms

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
                        # if totalNumberOfFalseAlarms - originalNumberOfFalseAlarms > 2:
                        print("   ", end="")
                        for element in actualVoiceActivity:
                            if element == 0:
                                print("-", end="")
                            else:
                                print("█", end="")
                        print()
                        print("   ", end="")
                        for element in voiceActivity:
                            if element == 0:
                                print("-", end="")
                            else:
                                print("█", end="")
                        print()

                            # print("   ", actualVoiceActivity)
                            # print("   ", voiceActivity)
                    else:
                        print(name, "✓")

                    # print("   ", sum(actualVoiceActivity))

                    totalNumberOfVoiceActivityInstances += int(sum(actualVoiceActivity))

    precision = totalNumberOfCorrectlyDetectedVoiceActivityInstances / (totalNumberOfCorrectlyDetectedVoiceActivityInstances + totalNumberOfFalseAlarms)
    recall = totalNumberOfCorrectlyDetectedVoiceActivityInstances / totalNumberOfVoiceActivityInstances

    fMeasure = 2 * precision * recall / (precision + recall)

    print("    Time      |", np.mean(times))

    print("   Actual     | Seconds with voice activity:", totalNumberOfVoiceActivityInstances)
    print("  Algorithm   | Correct detectsions:", totalNumberOfCorrectlyDetectedVoiceActivityInstances, "False alarms:", totalNumberOfFalseAlarms, "Precision:", precision, "Recall:", recall, "F-measure", fMeasure)


def validateOnRandomValidationSetsWithRVAD():
    validationTopLevelPath = "../media/validation/"

    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    print("name\t", "praatSyllableCount", "syllableCount")

    rVAD = []
    algorithm = []

    # Iterate through sub directories
    for validationSetPath in sorted(glob.iglob(validationTopLevelPath + '*/')):

        print(validationSetPath)

        totalNumberOfVoiceActivityInstances = 0
        totalNumberOfCorrectlyDetectedVoiceActivityInstances = 0
        totalNumberOfFalseAlarms = 0
        totalNumberOfCorrectRejections = 0
        totalInstances = 0

        for filePath in sorted(glob.iglob(validationSetPath + "*.wav")):
            fileName = os.path.basename(filePath)[:-4]

            audio = audioModule.Audio(filePath=filePath)
            if audio.numberOfChannels != 1:
                audio.makeMono()

            algorithmVoiceActivity = speechAnalyzer.getVoiceActivityFromAudio(audio)
            rVADVoiceActivity = voiceActivityFromRVAD(filePath)
            frameSizeInSeconds = 1
            frameSizeInSteps = int(frameSizeInSeconds / (speechAnalyzer.featureStepSize / 1000))

            algorithmVoiceActivityBins = []
            rVADVoiceActivityBins = []

            originalNumberOfFalseAlarms = totalNumberOfFalseAlarms

            for frame in range (0, int(len(audio.data) / audio.sampleRate / (frameSizeInSeconds))):
                start = frame * frameSizeInSteps
                end = frame * frameSizeInSteps + frameSizeInSteps

                algorithmVoiceActivityValue = int(max(algorithmVoiceActivity[start:end]))
                algorithmVoiceActivityBins.append(algorithmVoiceActivityValue)

                rVADVoiceActivityValue = int(max(rVADVoiceActivity[start:end]))
                rVADVoiceActivityBins.append(rVADVoiceActivityValue)

                if algorithmVoiceActivityValue == 1:
                    if rVADVoiceActivityValue == 0:
                        totalNumberOfFalseAlarms += 1
                    if rVADVoiceActivityValue == 1:
                        totalNumberOfCorrectlyDetectedVoiceActivityInstances += 1
                else:
                    if rVADVoiceActivityValue == 0:
                        totalNumberOfCorrectRejections += 1

            print(fileName)
            print("      rVAD ", end="")
            for element in rVADVoiceActivityBins:
                if element == 0:
                    print("-", end="")
                else:
                    print("█", end="")
            print()
            print(" algorithm ", end="")
            for element in algorithmVoiceActivityBins:
                if element == 0:
                    print("-", end="")
                else:
                    print("█", end="")
            print()

            algorithm.append(algorithmVoiceActivityBins)
            rVAD.append(rVADVoiceActivityBins)

            totalInstances += len(rVADVoiceActivityBins)
            totalNumberOfVoiceActivityInstances += int(sum(rVADVoiceActivityBins))

        precision = totalNumberOfCorrectlyDetectedVoiceActivityInstances / (totalNumberOfCorrectlyDetectedVoiceActivityInstances + totalNumberOfFalseAlarms)
        recall = totalNumberOfCorrectlyDetectedVoiceActivityInstances / totalNumberOfVoiceActivityInstances
        accuracy = (totalNumberOfCorrectlyDetectedVoiceActivityInstances + totalNumberOfCorrectRejections) / totalInstances

        fMeasure = 2 * precision * recall / (precision + recall)

        print("   This Set   | Seconds with voice activity:", totalNumberOfVoiceActivityInstances, "Total seconds:", totalInstances)
        print("  Algorithm   | Correct detectsions:", totalNumberOfCorrectlyDetectedVoiceActivityInstances, "False alarms:", totalNumberOfFalseAlarms, "Precision:", precision, "Recall:", recall, "F-measure:", fMeasure, "Accuracy:", accuracy)

def voiceActivityFromRVAD(wavFile):
    return rVAD_fast.voiceActivity(finwav= wavFile, vadThres= 1.2)

def main():
    validateOnRandomValidationSetsWithRVAD()

main()
