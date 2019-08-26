#
# Created on August 1, 2019
#
# @author: Julian Fortune
# @Description: Validatino of syllable algorithm.
#

import sys, time, glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
import parselmouth

from speechLibrary import featureModule, speechAnalysis, audioModule

np.set_printoptions(threshold=sys.maxsize)

def validateUsingTranscript():
    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    transcript = []

    with open(audioDirectory + "/syllables.txt") as transcriptFile:
        lines = transcriptFile.readlines()
        for row in lines:
            transcript.append(row.strip().split(', '))

    totalNumberOfSyllables = 0
    totalNumberOfCorrectlyDetectedSyllables = 0
    ourFalseAlarms = 0

    times = []

    for line in transcript:
        name = line[0]

        if name[0] != "#":
            actualSyllableCount = int(line[2])

            for filePath in sorted(glob.iglob(audioDirectory + "*.wav")):
                fileName = os.path.basename(filePath)[:-4]

                if fileName == name:
                    audio = audioModule.Audio(filePath=filePath)
                    if audio.numberOfChannels != 1:
                        audio.makeMono()


                    pitches = speechAnalyzer.getPitchFromAudio(audio)

                    startTime = time.time()
                    syllables, _ = speechAnalyzer.getSyllablesFromAudio(audio, pitches= pitches)
                    times.append(time.time() - startTime)

                    if True:
                        voiceActivity = speechAnalyzer.getVoiceActivityFromAudio(audio)
                        bufferFrames = int(speechAnalyzer.voiceActivityMaskBufferSize / speechAnalyzer.featureStepSize)
                        mask = np.invert(featureModule.createBufferedBinaryArrayFromArray(voiceActivity.astype(bool), bufferFrames))
                        syllables[mask] = 0

                    syllableCount = int(sum(syllables))

                    print(name, "\t", actualSyllableCount, syllableCount)

                    totalNumberOfSyllables += actualSyllableCount

                    if syllableCount > actualSyllableCount:
                        totalNumberOfFalseAlarms += syllableCount - actualSyllableCount
                        totalNumberOfCorrectlyDetectedSyllables += actualSyllableCount
                    else:
                        totalNumberOfCorrectlyDetectedSyllables += syllableCount

    precision = totalNumberOfCorrectlyDetectedSyllables / (totalNumberOfCorrectlyDetectedSyllables + totalNumberOfFalseAlarms)
    recall = totalNumberOfCorrectlyDetectedSyllables / totalNumberOfSyllables

    fMeasure = 2 * precision * recall / (precision + recall)

    print("    Time      |", np.mean(times))

    print("    Total     | Syllables:", totalNumberOfSyllables)
    print("     New      | Correct syllables:", totalNumberOfCorrectlyDetectedSyllables,
          "False alarms:", totalNumberOfFalseAlarms,
          "Precision:", precision, "Recall:", recall, "F1", fMeasure)

def validateUsingPRAATScript():
    validationTopLevelPath = "../media/validation/"

    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    print("name\t", "praatSyllableCount", "syllableCount")

    praat = []
    algorithm = []

    # Iterate through sub directories
    for validationSetPath in sorted(glob.iglob(validationTopLevelPath + '*/')):

        praatOutput = []

        with open(validationSetPath + "praat_script_output.txt") as transcriptFile:
            lines = transcriptFile.readlines()
            for row in lines:
                praatOutput.append(row.strip().split(', '))

        # Remove the header row
        praatOutput.pop(0)

        # Clean up the data a bit
        for index in range(0, len(praatOutput)):
            oringinalFileName = '_'.join(praatOutput[index][0].split('_')[:3]) + '.' + '.'.join(praatOutput[index][0].split('_')[3:])

            praatOutput[index] = praatOutput[index][:2]
            praatOutput[index][1] = int(praatOutput[index][1])
            praatOutput[index][0] = oringinalFileName

        for filePath in sorted(glob.iglob(validationSetPath + "*.wav")):
            fileName = os.path.basename(filePath)[:-4]

            existsInPraat = False

            for sample in praatOutput:
                name = sample[0]

                if name == fileName:
                    existsInPraat = True

                    praatSyllableCount = sample[1]

                    audio = audioModule.Audio(filePath=filePath)
                    audio.makeMono()

                    pitches = speechAnalyzer.getPitchFromAudio(audio)
                    syllables, _ = speechAnalyzer.getSyllablesFromAudio(audio, pitches= pitches)

                    if True:
                        voiceActivity = speechAnalyzer.getVoiceActivityFromAudio(audio, pitches= pitches)
                        bufferFrames = int(speechAnalyzer.voiceActivityMaskBufferSize / speechAnalyzer.featureStepSize)
                        mask = np.invert(featureModule.createBufferedBinaryArrayFromArray(voiceActivity.astype(bool), bufferFrames))
                        syllables[mask] = 0

                    syllableCount = int(sum(syllables))

                    praat.append(praatSyllableCount)
                    algorithm.append(syllableCount)

                    print('_'.join(name.split('_')[:2]), "\t", praatSyllableCount, syllableCount)

            if not existsInPraat:
                print("WARNING: Couldn't find " + fileName + " in PRAAT output.")

    print(praat, algorithm)

    plt.figure(figsize=[16, 8])
    plt.plot(praat, algorithm, 'o', alpha=0.15, label="30 second audio sample")
    plt.title('Comparison between PRAAT and the syllable algorithm')
    plt.ylabel('Syllables detected by our algorithm')
    plt.xlabel('Syllables detected by PRAAT script')
    plt.legend(loc='upper left')
    plt.show()

def runPRAATScript(directory, threshold=-35, intensityDip=8, captureOutput=True):
    _, result = parselmouth.praat.run_file("../validation/PRAAT/Praat Script Syllable Nuclei v2", threshold, intensityDip, 0.3, False, directory, capture_output= captureOutput)

    result = result.strip().split('\n')
    for i in range(0, len(result)):
        result[i] = result[i].split(', ')

    return result

def validateOnRandomValidationSetsWithPRAATScript():
    validationTopLevelPath = "../media/validation/"

    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    syllablesData = pd.DataFrame({}, columns = ["validationSet", "participant", "condition", "times", "praat", "ourAlgorithm"])

    praat = []
    algorithm = []

    praatDetectionsTotals = []
    ourCorrectDetectionsTotals = []
    ourFalseAlarmsTotals = []

    # Iterate through sub directories
    for validationSetPath in sorted(glob.iglob(validationTopLevelPath + '*/')):
        validationSet = validationSetPath.split('/')[-2]

        print()
        print(validationSetPath)
        print()

        praatDetections = 0
        ourCorrectDetections = 0
        ourFalseAlarms = 0

        print("Threshold: -35, intensityDip: 8")
        # print("Threshold: -30, intensityDip: 4")
        # print("Threshold: -25, intensityDip: 2")

        praatOutput = runPRAATScript(validationSetPath,
                                     threshold=-35,
                                     intensityDip=8)

        # Remove the header row
        praatOutput.pop(0)

        # Make new container for PRAAT syllable data
        praatData = []

        # Clean up the data a bit
        for index in range(0, len(praatOutput)):
            oringinalFileName = '_'.join(praatOutput[index][0].split('_')[:3]) + '.' + '.'.join(praatOutput[index][0].split('_')[3:])

            praatOutput[index][0] = oringinalFileName
            praatOutput[index][1] = int(praatOutput[index][1].replace(",", ""))

            praatOutput[index] = praatOutput[index][:2 + praatOutput[index][1]]

            syllables = []
            for subIndex in range(0, praatOutput[index][1]):
                syllables.append(float(praatOutput[index][2 + subIndex].replace(",", "")))

            praatData.append([oringinalFileName, syllables])

        for filePath in sorted(glob.iglob(validationSetPath + "*.wav")):
            fileName = os.path.basename(filePath)[:-4]

            participant = fileName.split('_')[0][1:]
            condition = fileName.split('_')[1]
            times = fileName.split('_')[2]

            existsInPraat = False

            for sample in praatData:
                name = sample[0]

                if name == fileName:
                    existsInPraat = True

                    praatTimeStamps = sample[1]

                    audio = audioModule.Audio(filePath=filePath)
                    audio.makeMono()

                    pitches = speechAnalyzer.getPitchFromAudio(audio)
                    syllables, _ = speechAnalyzer.getSyllablesFromAudio(audio, pitches= pitches)

                    praatSyllables = np.full(len(syllables), 0)

                    for time in praatTimeStamps:
                        praatSyllables[int(time / (speechAnalyzer.featureStepSize / 1000))] = 1

                    if True:
                        voiceActivity = speechAnalyzer.getVoiceActivityFromAudio(audio, pitches= pitches)

                        bufferFrames = int(speechAnalyzer.voiceActivityMaskBufferSize / speechAnalyzer.featureStepSize)
                        mask = np.invert(featureModule.createBufferedBinaryArrayFromArray(voiceActivity.astype(bool), bufferFrames))
                        syllables[mask] = 0
                        praatSyllables[mask] = 0

                    timeStamps = []
                    praatVoiceActivityTimeStamps = []

                    for syllableInstanceIndex in range(0, len(syllables)):
                        if syllables[syllableInstanceIndex] == 1:
                            timeStamps.append(syllableInstanceIndex * speechAnalyzer.featureStepSize / 1000)
                        if praatSyllables[syllableInstanceIndex] == 1:
                            praatVoiceActivityTimeStamps.append(syllableInstanceIndex * speechAnalyzer.featureStepSize / 1000)

                    if False: # Mask praat with our voice activity
                        praatTimeStamps = praatVoiceActivityTimeStamps

                    praatSyllableCount = len(praatTimeStamps)

                    praatDetections += praatSyllableCount

                    ogCorrect = ourCorrectDetections
                    ogFalseAlarms = ourFalseAlarms

                    timeStampsToMatch = timeStamps.copy()
                    matchingAlgorithmTimeStamps = 0

                    for timeStamp in praatTimeStamps:
                        distances = [abs(element - timeStamp) for element in timeStampsToMatch]

                        if len(distances) > 0:

                            indexOfClosestAlgorithmTimeStamp = distances.index(min(distances))

                            # Check distance is within threshold and not claimed already
                            if min(distances) < 0.1:
                                timeStampsToMatch.pop(indexOfClosestAlgorithmTimeStamp)
                                matchingAlgorithmTimeStamps += 1

                    ourCorrectDetections += matchingAlgorithmTimeStamps
                    ourFalseAlarms += len(timeStamps) - matchingAlgorithmTimeStamps

                    # Save syllable arrays.
                    syllablesData = syllablesData.append(pd.Series([validationSet, participant, condition, times, sample[1], timeStamps], index=syllablesData.columns ), ignore_index=True)

                    print('_'.join(name.split('_')[:2]), "\t", praatSyllableCount, ourCorrectDetections - ogCorrect, ourFalseAlarms - ogFalseAlarms)

            if not existsInPraat:
                print("WARNING: Couldn't find " + fileName + " in PRAAT output.")

        # Keep track of accuracy metrics
        praatDetectionsTotals.append(praatDetections)
        ourCorrectDetectionsTotals.append(ourCorrectDetections)
        ourFalseAlarmsTotals.append(ourFalseAlarms)

    # Add on a column with the total
    praatDetectionsTotals.append(sum(praatDetectionsTotals))
    ourCorrectDetectionsTotals.append(sum(ourCorrectDetectionsTotals))
    ourFalseAlarmsTotals.append(sum(ourFalseAlarmsTotals))

    ourDetectionsTotals = list(np.add(ourCorrectDetectionsTotals, ourFalseAlarmsTotals))

    # Print out all the crap
    print("praatTotals:                 ", praatDetectionsTotals, "\t", " & ".join(str(e) for e in praatDetectionsTotals))

    print("ourTotals:                  ", ourDetectionsTotals, "\t", " & ".join(str(e) for e in ourDetectionsTotals))
    print("ourCorrectDetectionsTotals: ", ourCorrectDetectionsTotals, "\t", " & ".join(str(e) for e in ourCorrectDetectionsTotals))
    print("ourFalseAlarmsTotals:       ", ourFalseAlarmsTotals, "\t", " & ".join(str(e) for e in ourFalseAlarmsTotals))

    ourPrecision = [round(e, 2) for e in np.divide(ourCorrectDetectionsTotals, np.add(ourCorrectDetectionsTotals, ourFalseAlarmsTotals))]
    ourRecall = [round(e, 2) for e in np.divide(ourCorrectDetectionsTotals, praatDetectionsTotals)]
    ourFMeasure = [round(e, 2) for e in 2 * np.divide(np.multiply(ourPrecision, ourRecall), np.add(ourPrecision, ourRecall))]

    print("ourPrecision:               ", ourPrecision, "\t", " & ".join(str(e) for e in ourPrecision))
    print("ourRecall:                  ", ourRecall, "\t", " & ".join(str(e) for e in ourRecall))
    print("ourFMeasure:                ", ourFMeasure, "\t", " & ".join(str(e) for e in ourFMeasure))

    syllablesData.to_csv("../validation/results/syllablesVoiceActivity.csv", index= False)



def main():
    validateOnRandomValidationSetsWithPRAATScript()

main()
