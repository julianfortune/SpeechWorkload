#
# Created on June 27, 2019
#
# @author: Julian Fortune
# @Description: Functions for testing and validating the syllable algorithm.
#

import sys, time, glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import librosa
import parselmouth

from speechLibrary import featureModule, speechAnalysis, audioModule

np.set_printoptions(threshold=sys.maxsize)

def comparePRAATSyllablesToPNNC():
    audioDirectory = "../media/pnnc-v1/audio/*.wav"
    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    transcript = []
    praatResults = []

    totalNumberOfFilledPauses = 0
    totalNumberOfCorrectlyDetectedPauses = 0
    totalNumberOfFalseAlarms = 0

    totalNumberOfAudioFiles = 0

    with open("../media/pnnc-v1/PNNC-transcripts.txt") as transcriptFile:
        lines = transcriptFile.readlines()
        for row in lines:
            transcript.append(row.strip().split('\t'))

    with open("../media/pnnc-v1/praat.txt") as praatFile:
        lines = praatFile.readlines()
        for row in lines:
            praatResults.append(row.strip().split(', '))

    # Remove column labels row
    praatResults.pop(0)

    for result in praatResults:
        name = result[0]
        sentenceReadByParticipant = name[-5:]
        syllableCount = int(result[1])

        for sentence in transcript:
            if sentenceReadByParticipant == sentence[0]:
                if len(sentence) > 2: # Check syllable count exists
                    actualSyllableCount = int(sentence[2])

                    totalNumberOfAudioFiles += 1
                    totalNumberOfFilledPauses += actualSyllableCount

                    if syllableCount > actualSyllableCount:
                        totalNumberOfFalseAlarms += syllableCount - actualSyllableCount
                        totalNumberOfCorrectlyDetectedPauses += actualSyllableCount
                    else:
                        totalNumberOfCorrectlyDetectedPauses += syllableCount

    precision = totalNumberOfCorrectlyDetectedPauses / (totalNumberOfCorrectlyDetectedPauses + totalNumberOfFalseAlarms)
    recall = totalNumberOfCorrectlyDetectedPauses / totalNumberOfFilledPauses

    f1 = 2 * precision * recall / (precision + recall)

    print("    Total     | Audio files", totalNumberOfAudioFiles, "Syllables:", totalNumberOfFilledPauses)
    print("    PRAAT     | Correct syllables:", totalNumberOfCorrectlyDetectedPauses, "Precision:", precision, "Recall:", recall, "F1", f1)

def compareSyllablesToPNNC():
    audioDirectory = "../media/pnnc-v1/audio/*.wav"
    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    transcript = []

    totalNumberOfFilledPauses = 0
    totalNumberOfCorrectlyDetectedPauses = 0
    totalNumberOfFalseAlarms = 0

    totalNumberOfAudioFiles = 0

    with open("../media/pnnc-v1/PNNC-transcripts.txt") as transcriptFile:
        lines = transcriptFile.readlines()
        for row in lines:
            transcript.append(row.strip().split('\t'))

    for sentence in transcript:
        if len(sentence) > 2: # Check syllable count exists
            for filePath in sorted(glob.iglob(audioDirectory)):
                fileName = os.path.basename(filePath)[:-4]
                sentenceReadByParticipant = fileName[-5:]

                if sentenceReadByParticipant == sentence[0]:
                    correctNumberOfFilledPauses = int(sentence[2])

                    audio = audioModule.Audio(filePath=filePath)
                    if audio.numberOfChannels != 1:
                        audio.makeMono()

                    syllables, timeStamps = speechAnalyzer.getSyllablesFromAudio(audio)

                    totalNumberOfAudioFiles += 1

                    totalNumberOfFilledPauses += correctNumberOfFilledPauses

                    if len(timeStamps) > correctNumberOfFilledPauses:
                        totalNumberOfFalseAlarms += len(timeStamps) - correctNumberOfFilledPauses
                        totalNumberOfCorrectlyDetectedPauses += correctNumberOfFilledPauses
                    else:
                        totalNumberOfCorrectlyDetectedPauses += len(timeStamps)

                    # syllableMarkers = np.full(len(pitchSyllables), 0)
                    # candidateMarkers = np.full(len(candidates), 0)
                    #
                    # ### Energy
                    # energy = librosa.feature.rmse(audio.data, frame_length=int(audio.sampleRate / 1000 * speechAnalyzer.featureStepSize), hop_length=int(audio.sampleRate / 1000 * speechAnalyzer.featureStepSize))[0]
                    # energyTimes = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.featureStepSize/1000)[:len(energy)]
                    #
                    # pitch = featureModule.getPitchAC(audio.data, audio.sampleRate, speechAnalyzer.featureStepSize)
                    # # pitchAdjustment = np.full(int(speechAnalyzer.syllableWindowSize/speechAnalyzer.featureStepSize - 1), np.nan)
                    # # pitch = np.append(pitchAdjustment, pitch)
                    # pitchTimes = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.featureStepSize/1000)[:len(pitch)]
                    #
                    # signalTimes = np.arange(0, len(audio.data)/audio.sampleRate, 1 / audio.sampleRate)
                    #
                    # plt.figure(figsize=[16, 8])
                    # plt.plot(energyTimes, energy / 10, pitchTimes, pitch, candidates, candidateMarkers, 'ro')
                    # plt.plot(pitchSyllables, syllableMarkers, 'go')
                    # plt.title(fileName + " | " + sentence[1])
                    # plt.show()

    pitchPrecision = totalNumberOfCorrectlyDetectedPauses / (totalNumberOfCorrectlyDetectedPauses + totalNumberOfFalseAlarms)
    pitchRecall = totalNumberOfCorrectlyDetectedPauses / totalNumberOfFilledPauses
    pitchF1 = 2 * pitchPrecision * pitchRecall / (pitchPrecision + pitchRecall)

    print("    Total     | Audio files", totalNumberOfAudioFiles, "Syllables:", totalNumberOfFilledPauses)
    print("   Alogrithm  | Correct syllables:", totalNumberOfCorrectlyDetectedPauses, "Precision:", pitchPrecision,"Recall:", pitchRecall, "F1", pitchF1)
    # print("Time just spent on pitch:", timeJustToGetPitches)

def compareAlgorithmAndPRAATWithPNNC():
    audioDirectory = "../media/pnnc-v1/audio/*.wav"
    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    syllablesFromEach = []

    transcript = []

    with open("../media/pnnc-v1/PNNC-transcripts.txt") as transcriptFile:
        lines = transcriptFile.readlines()
        for row in lines:
            transcript.append(row.strip().split('\t'))

    print(audioDirectory[:-6])

    praatResults = runPRAATScript(audioDirectory[:-6],
                                  threshold=-35,
                                  intensityDip=8)

    # Remove column labels row
    praatResults.pop(0)

    totalNumberOfAudioFiles = 0

    with open("../media/pnnc-v1/PNNC-transcripts.txt") as transcriptFile:
        lines = transcriptFile.readlines()
        for row in lines:
            transcript.append(row.strip().split('\t'))

    for sentence in transcript:
        if len(sentence) > 2: # Check syllable count exists

            correctNumberOfFilledPauses = 0
            algorithmNumberOfFilledPauses = 0
            praatNumberOfFilledPauses = 0

            for filePath in sorted(glob.iglob(audioDirectory)):
                fileName = os.path.basename(filePath)[:-4]
                sentenceReadByParticipant = fileName[-5:]

                if sentenceReadByParticipant == sentence[0]:
                    correctNumberOfFilledPauses = int(sentence[2])

                    audio = audioModule.Audio(filePath=filePath)
                    if audio.numberOfChannels != 1:
                        audio.makeMono()

                    syllables, timeStamps = speechAnalyzer.getSyllablesFromAudio(audio)

                    algorithmNumberOfFilledPauses = len(timeStamps)

            for result in praatResults:
                name = result[0]
                sentenceReadByParticipant = name[-5:]
                syllableCount = int(result[1])

                if sentenceReadByParticipant == sentence[0]:
                    if len(sentence) > 2: # Check syllable count exists
                        praatNumberOfFilledPauses = int(result[1])

            syllablesFromEach.append([correctNumberOfFilledPauses, praatNumberOfFilledPauses, algorithmNumberOfFilledPauses])

    print(syllablesFromEach)

    for row in syllablesFromEach:
        print(row[0], row[1], row[2])

def compareSyllablesToParticipants():
    audioDirectory = "../media/Participant_Audio_30_Sec_Chunks/*.wav"
    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    transcript = []

    totalNumberOfFilledPauses = 0
    totalNumberOfCorrectlyDetectedPauses = 0
    totalNumberOfFalseAlarms = 0

    with open("../media/Participant_Audio_30_Sec_Chunks_Transcripts/transcripts.txt") as transcriptFile:
        lines = transcriptFile.readlines()
        for row in lines:
            transcript.append(row.strip().split(', '))


    for line in transcript:
        name = line[0]
        actualSyllableCount = int(line[2])

        for filePath in sorted(glob.iglob(audioDirectory)):
            fileName = os.path.basename(filePath)[:-4]

            if fileName == name:
                audio = audioModule.Audio(filePath=filePath)
                if audio.numberOfChannels != 1:
                    audio.makeMono()

                syllables, timeStamps = speechAnalyzer.getSyllablesFromAudio(audio)
                syllableCount = len(timeStamps)

                print(name, actualSyllableCount, syllableCount)

                totalNumberOfFilledPauses += actualSyllableCount

                if syllableCount > actualSyllableCount:
                    totalNumberOfFalseAlarms += syllableCount - actualSyllableCount
                    totalNumberOfCorrectlyDetectedPauses += actualSyllableCount
                else:
                    totalNumberOfCorrectlyDetectedPauses += syllableCount

                # ### Energy
                # energy = librosa.feature.rmse(audio.data, frame_length=int(audio.sampleRate / 1000 * speechAnalyzer.featureStepSize), hop_length=int(audio.sampleRate / 1000 * speechAnalyzer.featureStepSize))[0]
                # energyTimes = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.featureStepSize/1000)[:len(energy)]
                #
                # energyMinThreshold = featureModule.getEnergyMinimumThreshold(energy)
                # fractionEnergyMinThreshold = energyMinThreshold / max(energy)
                #
                # pitch = featureModule.getPitchAC(audio.data, audio.sampleRate, speechAnalyzer.featureStepSize, fractionEnergyMinThreshold)
                # # pitchAdjustment = np.full(int(speechAnalyzer.syllableWindowSize/speechAnalyzer.featureStepSize - 1), np.nan)
                # # pitch = np.append(pitchAdjustment, pitch)
                # pitchTimes = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.featureStepSize/1000)[:len(pitch)]
                # syllableMarkers = np.full(len(syllables), 0)
                #
                # zcr = librosa.feature.zero_crossing_rate(audio.data, frame_length=int(audio.sampleRate / 1000 * speechAnalyzer.featureStepSize * 4), hop_length=int(audio.sampleRate / 1000 * speechAnalyzer.featureStepSize))[0]
                # zcrTimes = np.arange(0, len(audio.data)/audio.sampleRate + 1, speechAnalyzer.featureStepSize/1000)[:len(zcr)]
                #
                # plt.figure(figsize=[16, 8])
                # plt.plot(energyTimes, energy / 10, zcrTimes, zcr * 100, pitchTimes, pitch, syllables, syllableMarkers, 'go')
                # plt.title(fileName + " | " + line[1])
                # plt.show()

    precision = totalNumberOfCorrectlyDetectedPauses / (totalNumberOfCorrectlyDetectedPauses + totalNumberOfFalseAlarms)
    recall = totalNumberOfCorrectlyDetectedPauses / totalNumberOfFilledPauses

    f1 = 2 * precision * recall / (precision + recall)

    print("    Total     | Syllables:", totalNumberOfFilledPauses)
    print("     New      | Correct syllables:", totalNumberOfCorrectlyDetectedPauses, "Precision:", precision, "Recall:", recall, "F1", f1)

def tunePRAATScript():
    thresholdValues = range(-40, -32, 2)
    print("thresholdValues:", thresholdValues)

    intensityDipValues = range(6, 11, 1)

    audioDirectory = "../media/validation_participant_audio/"
    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    results = []

    for threshold in thresholdValues:
        for intensityDip in intensityDipValues:
            print(threshold, intensityDip)

            praatOutput = runPRAATScript(audioDirectory,
                                         threshold=threshold,
                                         intensityDip=intensityDip)

            # Remove the header row
            praatOutput.pop(0)

            # Clean up the data a bit
            for index in range(0, len(praatOutput)):
                oringinalFileName = '_'.join(praatOutput[index][0].split('_')[:3]) + '.' + '.'.join(praatOutput[index][0].split('_')[3:])

                praatOutput[index] = praatOutput[index][:2]
                praatOutput[index][1] = int(praatOutput[index][1])
                praatOutput[index][0] = oringinalFileName

            transcript = []

            with open(audioDirectory + "/syllables.txt") as transcriptFile:
                lines = transcriptFile.readlines()
                for row in lines:
                    transcript.append(row.strip().split(', '))

            totalNumberOfSyllables = 0
            totalNumberOfCorrectlyDetectedSyllables = 0
            totalNumberOfFalseAlarms = 0

            syllablesDetectedByPRAATDuringSilence = []

            for line in transcript:
                nameInTranscipt = line[0]

                if nameInTranscipt[0] != "#":
                    actualSyllableCount = int(line[2])

                    for sample in praatOutput:
                        name = sample[0]

                        if name == nameInTranscipt:
                            praatSyllableCount = sample[1]

                            # print(nameInTranscipt, actualSyllableCount, praatSyllableCount)

                            if actualSyllableCount == 0:
                                syllablesDetectedByPRAATDuringSilence.append(praatSyllableCount)

                            totalNumberOfSyllables += actualSyllableCount

                            if praatSyllableCount > actualSyllableCount:
                                totalNumberOfFalseAlarms += praatSyllableCount - actualSyllableCount
                                totalNumberOfCorrectlyDetectedSyllables += actualSyllableCount
                            else:
                                totalNumberOfCorrectlyDetectedSyllables += praatSyllableCount

            precision = totalNumberOfCorrectlyDetectedSyllables / (totalNumberOfCorrectlyDetectedSyllables + totalNumberOfFalseAlarms)
            recall = totalNumberOfCorrectlyDetectedSyllables / totalNumberOfSyllables

            fMeasure = 2 * precision * recall / (precision + recall)

            print("    Total     | Syllables:", totalNumberOfSyllables)
            print("    PRAAT     | Correct syllables:", totalNumberOfCorrectlyDetectedSyllables,
                  "False alarms:", totalNumberOfFalseAlarms,
                  "Precision:", precision, "Recall:", recall, "F1", fMeasure)
            print("    PRAAT     | Average syllables detected during silence:", np.mean(syllablesDetectedByPRAATDuringSilence))

            results.append([threshold, intensityDip, fMeasure])
        # --
    # --

    print(results)

def runPRAATScript(directory, threshold=-35, intensityDip=8, captureOutput=True):
    _, result = parselmouth.praat.run_file("../validation/PRAAT/Praat Script Syllable Nuclei v2", threshold, intensityDip, 0.3, False, directory, capture_output= captureOutput)

    result = result.strip().split('\n')
    for i in range(0, len(result)):
        result[i] = result[i].split(', ')

    return result

def compareAlgorithmToPraatScriptOnRandomDataSet():

    # Graph parameters
    validationColor = "mediumaquamarine"
    ourColor = "mediumpurple"
    plt.rc('font',**{'family':'serif','serif':['Palatino']})

    validationTopLevelPath = "../media/validation/"

    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    # Iterate through sub directories
    for validationSetPath in ["../media/validation/2/"]: #sorted(glob.iglob(validationTopLevelPath + '*/')):
        validationSet = validationSetPath.split('/')[-2]

        print()
        print(validationSetPath)
        print()

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

            print(participant, condition, times)

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

                    if True:
                        praatTimeStamps = praatVoiceActivityTimeStamps

                    praatValues = praatTimeStamps
                    ourAlgorithmValues = timeStamps

                    plt.figure(figsize=(11,3))
                    plt.plot(list(range(0, 30 + 1)), [0] * (30 + 1), drawstyle="steps", color="black")

                    for x_pos in praatValues:
                        plt.vlines(x_pos, ymin=0, ymax=1, color=validationColor)

                    for x_pos in ourAlgorithmValues:
                        plt.vlines(x_pos, ymin=-1, ymax=0, color=ourColor)

                    plt.title(fileName)
                    plt.ylim((-1.5, 1.5))
                    plt.xlim((0,30))
                    plt.box(on=None)
                    plt.subplots_adjust(top= 0.85, bottom= 0.25)
                    plt.yticks([])

                    patch1 = mpatches.Patch(color=validationColor)
                    patch2 = mpatches.Patch(color=ourColor)

                    patches = (patch1, patch2)

                    plt.legend(patches, ["Praat", "Ours"], loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2, )

                    outputPath = "../validation/comparison/syllables/" + validationSet + "/" + fileName + ".png"

                    # plt.show()
                    plt.savefig(outputPath)
                    plt.close()



def main():
    compareAlgorithmToPraatScriptOnRandomDataSet()

main()
