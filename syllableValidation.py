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
    totalNumberOfFalseAlarms = 0

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

    syllablesData = pd.DataFrame({}, columns = ["validationSet", "participant", "condition", "times", "rVAD", "ourAlgorithm"])

    praat = []
    algorithm = []

    # Iterate through sub directories
    for validationSetPath in sorted(glob.iglob(validationTopLevelPath + '*/')):
        validationSet = validationSetPath.split('/')[-2]

        print()
        print(validationSetPath)
        print()

        totalNumberOfSyllables = 0
        totalNumberOfCorrectlyDetectedSyllables = 0
        totalNumberOfFalseAlarms = 0

        praatOutput = runPRAATScript(validationSetPath,
                                     threshold=-35,
                                     intensityDip=8)

        # Remove the header row
        praatOutput.pop(0)

        # Make new container for PRAAT syllable data
        praatSyllables = []

        # Clean up the data a bit
        for index in range(0, len(praatOutput)):
            oringinalFileName = '_'.join(praatOutput[index][0].split('_')[:3]) + '.' + '.'.join(praatOutput[index][0].split('_')[3:])

            praatOutput[index][0] = oringinalFileName
            praatOutput[index][1] = int(praatOutput[index][1].replace(",", ""))

            praatOutput[index] = praatOutput[index][:2 + praatOutput[index][1]]

            syllables = []
            for subIndex in range(0, praatOutput[index][1]):
                syllables.append(float(praatOutput[index][2 + subIndex].replace(",", "")))

            praatSyllables.append([oringinalFileName, syllables])

        for filePath in sorted(glob.iglob(validationSetPath + "*.wav")):
            fileName = os.path.basename(filePath)[:-4]

            participant = fileName.split('_')[0][1:]
            condition = fileName.split('_')[1]
            times = fileName.split('_')[2]

            print(participant, condition, times)

            existsInPraat = False

            for sample in praatSyllables:
                name = sample[0]

                if name == fileName:
                    existsInPraat = True

                    praatSyllableCount = len(sample[1])

                    audio = audioModule.Audio(filePath=filePath)
                    audio.makeMono()

                    pitches = speechAnalyzer.getPitchFromAudio(audio)
                    syllables, _ = speechAnalyzer.getSyllablesFromAudio(audio, pitches= pitches)

                    if True:
                        voiceActivity = speechAnalyzer.getVoiceActivityFromAudio(audio, pitches= pitches)

                        bufferFrames = int(speechAnalyzer.voiceActivityMaskBufferSize / speechAnalyzer.featureStepSize)
                        mask = np.invert(featureModule.createBufferedBinaryArrayFromArray(voiceActivity.astype(bool), bufferFrames))
                        syllables[mask] = 0

                    timeStamps = []

                    for syllableInstanceIndex in range(0, len(syllables)):
                        if syllables[syllableInstanceIndex] == 1:
                            timeStamps.append(syllableInstanceIndex * speechAnalyzer.featureStepSize / 1000)

                    totalNumberOfSyllables += praatSyllableCount

                    ogCorrect = totalNumberOfCorrectlyDetectedSyllables
                    ogFalseAlarms = totalNumberOfFalseAlarms

                    for timeStamp in timeStamps:
                        correspondsToPRAATTimeStamp = False
                        for praatSyllable in sample[1]:
                            if abs(timeStamp - praatSyllable) < 0.1:
                                correspondsToPRAATTimeStamp = True

                        if correspondsToPRAATTimeStamp:
                            totalNumberOfCorrectlyDetectedSyllables += 1
                        else:
                            totalNumberOfFalseAlarms += 1

                    # Save syllable arrays.
                    syllablesData = syllablesData.append(pd.Series([validationSet, participant, condition, times, sample[1], timeStamps], index=syllablesData.columns ), ignore_index=True)

                    print('_'.join(name.split('_')[:2]), "\t", praatSyllableCount, totalNumberOfCorrectlyDetectedSyllables - ogCorrect, totalNumberOfFalseAlarms - ogFalseAlarms)

            if not existsInPraat:
                print("WARNING: Couldn't find " + fileName + " in PRAAT output.")

        precision = totalNumberOfCorrectlyDetectedSyllables / (totalNumberOfCorrectlyDetectedSyllables + totalNumberOfFalseAlarms)
        recall = totalNumberOfCorrectlyDetectedSyllables / totalNumberOfSyllables
        fMeasure = 2 * precision * recall / (precision + recall)

        print("   This Set   | Syllables:", totalNumberOfSyllables)
        print("  Algorithm   | Correct syllables:", totalNumberOfCorrectlyDetectedSyllables,
              "False alarms:", totalNumberOfFalseAlarms,
              "Precision:", precision, "Recall:", recall, "F1", fMeasure)

    syllablesData.to_csv("../validation/results/syllables.csv", index= False)

    #
    # # Total for all sets
    # totalNumberOfSyllables = 0
    # totalNumberOfCorrectlyDetectedSyllables = 0
    # totalNumberOfFalseAlarms = 0
    #
    # assert len(praat) == len(algorithm)
    # for indexOfAudioFileProcessed in range(0, len(praat)):
    #     actualSyllableCount = praat[indexOfAudioFileProcessed]
    #     algorithmSyllableCount = algorithm[indexOfAudioFileProcessed]
    #
    #     totalNumberOfSyllables += actualSyllableCount
    #
    #     if algorithmSyllableCount > actualSyllableCount:
    #         totalNumberOfFalseAlarms += algorithmSyllableCount - actualSyllableCount
    #         totalNumberOfCorrectlyDetectedSyllables += actualSyllableCount
    #     else:
    #         totalNumberOfCorrectlyDetectedSyllables += algorithmSyllableCount
    #
    # precision = totalNumberOfCorrectlyDetectedSyllables / (totalNumberOfCorrectlyDetectedSyllables + totalNumberOfFalseAlarms)
    # recall = totalNumberOfCorrectlyDetectedSyllables / totalNumberOfSyllables
    # fMeasure = 2 * precision * recall / (precision + recall)
    #
    # print("    Total     | Syllables:", totalNumberOfSyllables)
    # print("  Algorithm   | Correct syllables:", totalNumberOfCorrectlyDetectedSyllables,
    #       "False alarms:", totalNumberOfFalseAlarms,
    #       "Precision:", precision, "Recall:", recall, "F1", fMeasure)
    # if False:
    #     plt.figure(figsize=[16, 8])
    #     plt.plot(praat, algorithm, 'o', alpha=0.15, label="30 second audio sample")
    #     plt.title('Comparison between PRAAT and the syllable algorithm')
    #     plt.ylabel('Syllables detected by our algorithm')
    #     plt.xlabel('Syllables detected by PRAAT script')
    #     plt.legend(loc='upper left')
    #     plt.show()

def main():
    validateOnRandomValidationSetsWithPRAATScript()

main()
