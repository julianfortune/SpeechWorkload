#
# Created on July 30, 2019
#
# @author: Julian Fortune
# @Description:
#

import sys, time, glob, os
import numpy as np
import pandas as pd
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

def validateOnRandomValidationSetsWithRVAD(visuals= True):
    validationTopLevelPath = "../media/validation/"

    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    voiceActivityData = pd.DataFrame({}, columns = ["validationSet", "participant", "condition", "times", "rVAD", "ourAlgorithm"])

    rVAD = []
    algorithm = []

    rVADCorrectDetectionsTotals = []
    rVADFalseAlarmsTotals = []
    rVADCorrectRejectionsTotals = []

    ourCorrectDetectionsTotals = []
    ourFalseAlarmsTotals = []
    ourCorrectRejectionsTotals = []

    classificationTotals = []

    paths = [validationTopLevelPath + str(element) + '/' for element in list(range(1,11))]

    # Iterate through sub directories
    for validationSetPath in paths:
        validationSet = validationSetPath.split('/')[-2]

        print()
        print(validationSetPath)
        print()

        rVADCorrectDetections = 0
        rVADFalseAlarms = 0
        rVADCorrectRejections = 0

        ourCorrectDetections = 0
        ourFalseAlarms = 0
        ourCorrectRejections = 0

        totalClassifications = 0

        for filePath in sorted(glob.iglob(validationSetPath + "*.wav")):
            fileName = os.path.basename(filePath)[:-4]

            participant = fileName.split('_')[0][1:]
            condition = fileName.split('_')[1]
            times = fileName.split('_')[2]

            audio = audioModule.Audio(filePath=filePath)
            if audio.numberOfChannels != 1:
                audio.makeMono()

            algorithmVoiceActivity = speechAnalyzer.getVoiceActivityFromAudio(audio)
            rVADVoiceActivity = voiceActivityFromRVAD(filePath)
            frameSizeInSeconds = 1
            frameSizeInSteps = int(frameSizeInSeconds / (speechAnalyzer.featureStepSize / 1000))

            algorithmVoiceActivityBins = []
            rVADVoiceActivityBins = []

            for frame in range (0, int(len(audio.data) / audio.sampleRate / (frameSizeInSeconds))):
                start = frame * frameSizeInSteps
                end = frame * frameSizeInSteps + frameSizeInSteps

                algorithmVoiceActivityValue = int(max(algorithmVoiceActivity[start:end]))
                algorithmVoiceActivityBins.append(algorithmVoiceActivityValue)

                rVADVoiceActivityValue = int(max(rVADVoiceActivity[start:end]))
                rVADVoiceActivityBins.append(rVADVoiceActivityValue)

                if algorithmVoiceActivityValue == 1:
                    if rVADVoiceActivityValue == 1:
                        ourCorrectDetections += 1
                    else:
                        ourFalseAlarms += 1
                else:
                    if rVADVoiceActivityValue == 0:
                        ourCorrectRejections += 1

                if rVADVoiceActivityValue == 1:
                    if algorithmVoiceActivityValue == 1:
                        rVADCorrectDetections += 1
                    else:
                        rVADFalseAlarms += 1
                else:
                    if algorithmVoiceActivityValue == 0:
                        rVADCorrectRejections += 1

            totalClassifications += len(rVADVoiceActivityBins)

            # Save voice activity arrays.
            voiceActivityData = voiceActivityData.append(pd.Series([validationSet, participant, condition, times, rVADVoiceActivityBins, algorithmVoiceActivityBins], index=voiceActivityData.columns ), ignore_index=True)

            if visuals:
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

        rVADCorrectDetectionsTotals.append(rVADCorrectDetections)
        rVADFalseAlarmsTotals.append(rVADFalseAlarms)
        rVADCorrectRejectionsTotals.append(rVADCorrectRejections)

        ourCorrectDetectionsTotals.append(ourCorrectDetections)
        ourFalseAlarmsTotals.append(ourFalseAlarms)
        ourCorrectRejectionsTotals.append(ourCorrectRejections)

        classificationTotals.append(totalClassifications)

    # Add on a column with the total
    rVADCorrectDetectionsTotals.append(sum(rVADCorrectDetectionsTotals))
    rVADFalseAlarmsTotals.append(sum(rVADFalseAlarmsTotals))
    rVADCorrectRejectionsTotals.append(sum(rVADCorrectRejectionsTotals))

    ourCorrectDetectionsTotals.append(sum(ourCorrectDetectionsTotals))
    ourFalseAlarmsTotals.append(sum(ourFalseAlarmsTotals))
    ourCorrectRejectionsTotals.append(sum(ourCorrectRejectionsTotals))

    classificationTotals.append(sum(classificationTotals))

    rVADDetectionsTotals = list(np.add(rVADCorrectDetectionsTotals, rVADFalseAlarmsTotals))
    ourDetectionsTotals = list(np.add(ourCorrectDetectionsTotals, ourFalseAlarmsTotals))

    # Print out all the crap
    print("rVADTotals:                 ", rVADDetectionsTotals, "\t", " & ".join(str(e) for e in rVADDetectionsTotals))
    print("rVADCorrectDetectionsTotals:", rVADCorrectDetectionsTotals, "\t", " & ".join(str(e) for e in rVADCorrectDetectionsTotals))
    print("rVADFalseAlarmsTotals:      ", rVADFalseAlarmsTotals, "\t", " & ".join(str(e) for e in rVADFalseAlarmsTotals))
    print("rVADCorrectRejectionsTotals:", rVADCorrectRejectionsTotals, "\t", " & ".join(str(e) for e in rVADCorrectRejectionsTotals))

    rVADPrecision = [round(e, 2) for e in np.divide(rVADCorrectDetectionsTotals, np.add(rVADCorrectDetectionsTotals, rVADFalseAlarmsTotals))]
    rVADRecall = [round(e, 2) for e in np.divide(rVADCorrectDetectionsTotals, ourDetectionsTotals)]
    rVADAccuracy = [round(e, 2) for e in np.divide(np.add(rVADCorrectDetectionsTotals, rVADCorrectRejectionsTotals), classificationTotals)]
    rVADFMeasure = [round(e, 2) for e in 2 * np.divide(np.multiply(rVADPrecision, rVADRecall), np.add(rVADPrecision, rVADRecall))]

    print("rVADPrecision:              ", rVADPrecision, "\t", " & ".join(str(e) for e in rVADPrecision))
    print("rVADRecall:                 ", rVADRecall, "\t", " & ".join(str(e) for e in rVADRecall))
    print("rVADAccuracy:               ", rVADAccuracy, "\t", " & ".join(str(e) for e in rVADAccuracy))
    print("rVADFMeasure:               ", rVADFMeasure, "\t", " & ".join(str(e) for e in rVADFMeasure))

    print("ourTotals:                  ", ourDetectionsTotals, "\t", " & ".join(str(e) for e in ourDetectionsTotals))
    print("ourCorrectDetectionsTotals: ", ourCorrectDetectionsTotals, "\t", " & ".join(str(e) for e in ourCorrectDetectionsTotals))
    print("ourFalseAlarmsTotals:       ", ourFalseAlarmsTotals, "\t", " & ".join(str(e) for e in ourFalseAlarmsTotals))
    print("ourCorrectRejectionsTotals: ", ourCorrectRejectionsTotals, "\t", " & ".join(str(e) for e in ourCorrectRejectionsTotals))

    ourPrecision = [round(e, 2) for e in np.divide(ourCorrectDetectionsTotals, np.add(ourCorrectDetectionsTotals, ourFalseAlarmsTotals))]
    ourRecall = [round(e, 2) for e in np.divide(ourCorrectDetectionsTotals, rVADDetectionsTotals)]
    ourAccuracy = [round(e, 2) for e in np.divide(np.add(ourCorrectDetectionsTotals, ourCorrectRejectionsTotals), classificationTotals)]
    ourFMeasure = [round(e, 2) for e in 2 * np.divide(np.multiply(ourPrecision, ourRecall), np.add(ourPrecision, ourRecall))]

    print("ourPrecision:               ", ourPrecision, "\t", " & ".join(str(e) for e in ourPrecision))
    print("ourRecall:                  ", ourRecall, "\t", " & ".join(str(e) for e in ourRecall))
    print("ourAccuracy:                ", ourAccuracy, "\t", " & ".join(str(e) for e in ourAccuracy))
    print("ourFMeasure:                ", ourFMeasure, "\t", " & ".join(str(e) for e in ourFMeasure))

    print("classificationTotals:       ", classificationTotals, "\t", " & ".join(str(e) for e in classificationTotals))

    voiceActivityData.to_csv("../validation/results/voiceActivity.csv", index= False)

def voiceActivityFromRVAD(wavFile):
    return rVAD_fast.voiceActivity(finwav= wavFile, vadThres= 1.2)

def main():
    validateOnRandomValidationSetsWithRVAD(visuals= False)

main()
