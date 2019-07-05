import sys, time, glob, os
import numpy as np
import matplotlib.pyplot as plt

import librosa

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

                    totalNumberOfFilledPauses += actualSyllableCount

                    if syllableCount > actualSyllableCount:
                        totalNumberOfFalseAlarms += syllableCount - actualSyllableCount
                        totalNumberOfCorrectlyDetectedPauses += actualSyllableCount
                    else:
                        totalNumberOfCorrectlyDetectedPauses += syllableCount

    precision = totalNumberOfCorrectlyDetectedPauses / (totalNumberOfCorrectlyDetectedPauses + totalNumberOfFalseAlarms)
    recall = totalNumberOfCorrectlyDetectedPauses / totalNumberOfFilledPauses

    f1 = 2 * precision * recall / (precision + recall)

    print("    Total     | Syllables:", totalNumberOfFilledPauses)
    print("    PRAAT     | Correct syllables:", totalNumberOfCorrectlyDetectedPauses, "Precision:", precision, "Recall:", recall, "F1", f1)

def compareSyllablesToPNNC():
    audioDirectory = "../media/pnnc-v1/audio/*.wav"
    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    transcript = []

    totalNumberOfFilledPauses = 0
    totalNumberOfCorrectlyDetectedPauses = 0
    totalNumberOfFalseAlarms = 0

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

                    syllables, candidates = speechAnalyzer.getSyllablesFromAudio(audio)

                    totalNumberOfFilledPauses += correctNumberOfFilledPauses

                    if len(syllables) > correctNumberOfFilledPauses:
                        totalNumberOfFalseAlarms += len(syllables) - correctNumberOfFilledPauses
                        totalNumberOfCorrectlyDetectedPauses += correctNumberOfFilledPauses
                    else:
                        totalNumberOfCorrectlyDetectedPauses += len(syllables)

                    # syllableMarkers = np.full(len(pitchSyllables), 0)
                    # candidateMarkers = np.full(len(candidates), 0)
                    #
                    # ### Energy
                    # energy = librosa.feature.rmse(audio.data, frame_length=int(audio.sampleRate / 1000 * speechAnalyzer.syllableStepSize), hop_length=int(audio.sampleRate / 1000 * speechAnalyzer.syllableStepSize))[0]
                    # energyTimes = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.syllableStepSize/1000)[:len(energy)]
                    #
                    # pitch = featureModule.getPitchAC(audio.data, audio.sampleRate, speechAnalyzer.syllableStepSize)
                    # # pitchAdjustment = np.full(int(speechAnalyzer.syllableWindowSize/speechAnalyzer.syllableStepSize - 1), np.nan)
                    # # pitch = np.append(pitchAdjustment, pitch)
                    # pitchTimes = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.syllableStepSize/1000)[:len(pitch)]
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

    print("   Alogrithm  | Correct syllables:", totalNumberOfCorrectlyDetectedPauses, "Precision:", pitchPrecision,"Recall:", pitchRecall, "F1", pitchF1)
    # print("Time just spent on pitch:", timeJustToGetPitches)

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

                syllables, _ = speechAnalyzer.getSyllablesFromAudio(audio)
                syllableCount = len(syllables)

                print(name, actualSyllableCount, syllableCount)

                totalNumberOfFilledPauses += actualSyllableCount

                if syllableCount > actualSyllableCount:
                    totalNumberOfFalseAlarms += syllableCount - actualSyllableCount
                    totalNumberOfCorrectlyDetectedPauses += actualSyllableCount
                else:
                    totalNumberOfCorrectlyDetectedPauses += syllableCount

                # ### Energy
                # energy = librosa.feature.rmse(audio.data, frame_length=int(audio.sampleRate / 1000 * speechAnalyzer.syllableStepSize), hop_length=int(audio.sampleRate / 1000 * speechAnalyzer.syllableStepSize))[0]
                # energyTimes = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.syllableStepSize/1000)[:len(energy)]
                #
                # energyMinThreshold = featureModule.getEnergyMinimumThreshold(energy)
                # fractionEnergyMinThreshold = energyMinThreshold / max(energy)
                #
                # pitch = featureModule.getPitchAC(audio.data, audio.sampleRate, speechAnalyzer.syllableStepSize, fractionEnergyMinThreshold)
                # # pitchAdjustment = np.full(int(speechAnalyzer.syllableWindowSize/speechAnalyzer.syllableStepSize - 1), np.nan)
                # # pitch = np.append(pitchAdjustment, pitch)
                # pitchTimes = np.arange(0, len(audio.data)/audio.sampleRate, speechAnalyzer.syllableStepSize/1000)[:len(pitch)]
                # syllableMarkers = np.full(len(syllables), 0)
                #
                # zcr = librosa.feature.zero_crossing_rate(audio.data, frame_length=int(audio.sampleRate / 1000 * speechAnalyzer.syllableStepSize * 4), hop_length=int(audio.sampleRate / 1000 * speechAnalyzer.syllableStepSize))[0]
                # zcrTimes = np.arange(0, len(audio.data)/audio.sampleRate + 1, speechAnalyzer.syllableStepSize/1000)[:len(zcr)]
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

def main():
    comparePRAATSyllablesToPNNC()
    compareSyllablesToPNNC()
    # compareSyllablesToParticipants()

main()
