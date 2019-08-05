#
# Created on August 1, 2019
#
# @author: Julian Fortune
# @Description: Validatino of syllable algorithm.
#

import sys, time, glob, os
import numpy as np
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

def runPRAATScript(directory, threshold=-25, intensityDip=2, captureOutput=True):
    _, result = parselmouth.praat.run_file("../speechRateValidation/praat/Praat Script Syllable Nuclei v2", threshold, intensityDip, 0.3, False, directory, capture_output= captureOutput)

    result = result.strip().split('\n')
    for i in range(0, len(result)):
        result[i] = result[i].split(', ')

    return result

def syllablesUsingPRAATScript():
    validationTopLevelPath = "../media/validation/"

    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    praat = []
    algorithm = []

    # Iterate through sub directories
    for validationSetPath in sorted(glob.iglob(validationTopLevelPath + '*/')):

        print(validationSetPath)

        totalNumberOfSyllables = 0
        totalNumberOfCorrectlyDetectedSyllables = 0
        totalNumberOfFalseAlarms = 0

        praatOutput = runPRAATScript(validationSetPath,
                                     threshold=-35,
                                     intensityDip=8)

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

                    algorithmSyllableCount = int(sum(syllables))

                    totalNumberOfSyllables += praatSyllableCount

                    if algorithmSyllableCount > praatSyllableCount:
                        totalNumberOfFalseAlarms += algorithmSyllableCount - praatSyllableCount
                        totalNumberOfCorrectlyDetectedSyllables += praatSyllableCount
                    else:
                        totalNumberOfCorrectlyDetectedSyllables += algorithmSyllableCount

                    praat.append(praatSyllableCount)
                    algorithm.append(algorithmSyllableCount)

                    print('_'.join(name.split('_')[:2]), "\t", praatSyllableCount, algorithmSyllableCount)

            if not existsInPraat:
                print("WARNING: Couldn't find " + fileName + " in PRAAT output.")

        precision = totalNumberOfCorrectlyDetectedSyllables / (totalNumberOfCorrectlyDetectedSyllables + totalNumberOfFalseAlarms)
        recall = totalNumberOfCorrectlyDetectedSyllables / totalNumberOfSyllables
        fMeasure = 2 * precision * recall / (precision + recall)

        print("   This Set   | Syllables:", totalNumberOfSyllables)
        print("  Algorithm   | Correct syllables:", totalNumberOfCorrectlyDetectedSyllables,
              "False alarms:", totalNumberOfFalseAlarms,
              "Precision:", precision, "Recall:", recall, "F1", fMeasure)

    # Total for all sets
    totalNumberOfSyllables = 0
    totalNumberOfCorrectlyDetectedSyllables = 0
    totalNumberOfFalseAlarms = 0

    assert len(praat) == len(algorithm)
    for indexOfAudioFileProcessed in range(0, len(praat)):
        actualSyllableCount = praat[indexOfAudioFileProcessed]
        algorithmSyllableCount = algorithm[indexOfAudioFileProcessed]

        totalNumberOfSyllables += actualSyllableCount

        if algorithmSyllableCount > actualSyllableCount:
            totalNumberOfFalseAlarms += algorithmSyllableCount - actualSyllableCount
            totalNumberOfCorrectlyDetectedSyllables += actualSyllableCount
        else:
            totalNumberOfCorrectlyDetectedSyllables += algorithmSyllableCount

    precision = totalNumberOfCorrectlyDetectedSyllables / (totalNumberOfCorrectlyDetectedSyllables + totalNumberOfFalseAlarms)
    recall = totalNumberOfCorrectlyDetectedSyllables / totalNumberOfSyllables
    fMeasure = 2 * precision * recall / (precision + recall)

    print("    Total     | Syllables:", totalNumberOfSyllables)
    print("  Algorithm   | Correct syllables:", totalNumberOfCorrectlyDetectedSyllables,
          "False alarms:", totalNumberOfFalseAlarms,
          "Precision:", precision, "Recall:", recall, "F1", fMeasure)
    if False:
        plt.figure(figsize=[16, 8])
        plt.plot(praat, algorithm, 'o', alpha=0.15, label="30 second audio sample")
        plt.title('Comparison between PRAAT and the syllable algorithm')
        plt.ylabel('Syllables detected by our algorithm')
        plt.xlabel('Syllables detected by PRAAT script')
        plt.legend(loc='upper left')
        plt.show()

def main():
    syllablesUsingPRAATScript()

main()
