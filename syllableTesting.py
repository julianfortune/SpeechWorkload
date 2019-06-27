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
    shouldPrint = False

    transcript = []

    totalNumberOfFilledPauses = 0
    totalNumberOfCorrectlyDetectedPausesWithPitch = 0
    totalNumberOfFalseAlarmsWithPitch = 0

    totalNumberOfCorrectlyDetectedPausesWithoutPitch = 0
    totalNumberOfFalseAlarmsWithoutPitch = 0

    timeToRunWithPitches = 0
    timeToRunWithoutPitches = 0
    timeJustToGetPitches = 0

    with open("../media/pnnc-v1/PNNC-transcripts.txt") as transcriptFile:
        lines = transcriptFile.readlines()
        for row in lines:
            transcript.append(row.strip().split('\t'))
    if shouldPrint:
        print("Real | With Pitch | With zcr and fft")

    for sentence in transcript:
        if len(sentence) > 2: # Check syllable count exists
            if shouldPrint:
                print(sentence[0])
            for filePath in sorted(glob.iglob(audioDirectory)):
                fileName = os.path.basename(filePath)[:-4]
                sentenceReadByParticipant = fileName[-5:]

                if sentenceReadByParticipant == sentence[0]:
                    correctNumberOfFilledPauses = int(sentence[2])

                    audio = audioModule.Audio(filePath=filePath)
                    if audio.numberOfChannels != 1:
                        audio.makeMono()

                    startTime = time.time()
                    syllables, _ = speechAnalyzer.getSyllablesFromAudio(audio)
                    timeToRunWithoutPitches += time.time() - startTime

                    startTime = time.time()
                    pitchSyllables, candidates = speechAnalyzer.getSyllablesWithPitchFromAudio(audio)
                    timeToRunWithPitches += time.time() - startTime

                    startTime = time.time()
                    featureModule.getPitchAC(audio.data, audio.sampleRate, 10)
                    timeJustToGetPitches += time.time() - startTime

                    if shouldPrint:
                        print(correctNumberOfFilledPauses, "|", len(pitchSyllables), "|", len(syllables))



                    totalNumberOfFilledPauses += correctNumberOfFilledPauses

                    if len(pitchSyllables) > correctNumberOfFilledPauses:
                        totalNumberOfFalseAlarmsWithPitch += len(pitchSyllables) - correctNumberOfFilledPauses
                        totalNumberOfCorrectlyDetectedPausesWithPitch += correctNumberOfFilledPauses
                    else:
                        totalNumberOfCorrectlyDetectedPausesWithPitch += len(pitchSyllables)

                    if len(syllables) > correctNumberOfFilledPauses:
                        totalNumberOfFalseAlarmsWithoutPitch += len(syllables) - correctNumberOfFilledPauses
                        totalNumberOfCorrectlyDetectedPausesWithoutPitch += correctNumberOfFilledPauses
                    else:
                        totalNumberOfCorrectlyDetectedPausesWithoutPitch += len(syllables)

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

    pitchPrecision = totalNumberOfCorrectlyDetectedPausesWithPitch / (totalNumberOfCorrectlyDetectedPausesWithPitch + totalNumberOfFalseAlarmsWithPitch)
    pitchRecall = totalNumberOfCorrectlyDetectedPausesWithPitch / totalNumberOfFilledPauses
    pitchF1 = 2 * pitchPrecision * pitchRecall / (pitchPrecision + pitchRecall)

    print("    Pitch     | Correct syllables:", totalNumberOfCorrectlyDetectedPausesWithPitch, "Precision:", pitchPrecision,"Recall:", pitchRecall, "F1", pitchF1, "Time to run:", timeToRunWithPitches)
    print("Without Pitch | Correct syllables:", totalNumberOfCorrectlyDetectedPausesWithoutPitch, "Precision:", totalNumberOfCorrectlyDetectedPausesWithoutPitch / (totalNumberOfCorrectlyDetectedPausesWithoutPitch + totalNumberOfFalseAlarmsWithoutPitch),"Recall:", totalNumberOfCorrectlyDetectedPausesWithoutPitch / totalNumberOfFilledPauses, "Time to run:", timeToRunWithoutPitches)
    # print("Time just spent on pitch:", timeJustToGetPitches)


def main():
    comparePRAATSyllablesToPNNC()
    compareSyllablesToPNNC()

main()
