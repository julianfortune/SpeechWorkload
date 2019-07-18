#
# Created on July 11, 2019
#
# @author: Julian Fortune
# @Description: Functions for validating speech characteristics algorithms.
#

import sys, time, glob, os
import wavio
import csv

import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment

import random

from speechLibrary import featureModule, speechAnalysis, audioModule

np.set_printoptions(threshold=sys.maxsize)

audioDirectory = "../media/validation_participant_audio/"

# | Makes a 30-second segment from each audio file (30 participants x 3 conditions)
def createValidationSetFromParticipants():
    participantDirectory = "../media/Participant_Audio/*.wav"
    outputDir = "../media/filled_pauses_validation_participant_audio/"

    segmentsPerAudioFile = 3

    segmentLength = 30 * 1000 # In milliseconds

    for filePath in  sorted(glob.iglob(participantDirectory)):
        name = os.path.basename(filePath)[:-4]

        participant = name.split("_")[0]
        condition = name.split("_")[1]

        if participant == "p2" and condition == "ul":
            print("Skipping p2_ul")
        else:
            print(participant, condition)

            existingRanges = []

            for sampleFilePath in glob.iglob(audioDirectory + "*.wav"):
                sampleName = os.path.basename(sampleFilePath)[:-4]

                sampleParticipant = sampleName.split("_")[0]
                sampleCondition = sampleName.split("_")[1]

                if sampleParticipant == participant and sampleCondition == condition:
                    print(sampleName)

                    sampleStartTime = int(float(sampleName.split("_")[2].split("-")[0]) * 1000)
                    sampleEndTime = int(float(sampleName.split("_")[2].split("-")[1]) * 1000)

                    existingRanges.append([sampleStartTime, sampleEndTime])

            audio = AudioSegment.from_wav(filePath)
            audioObject = audioModule.Audio(filePath=filePath)
            audioObject.makeMono()

            length = int(len(audioObject.data) / audioObject.sampleRate * 1000)
            segmentStartRange = length - segmentLength

            for i in range(0, segmentsPerAudioFile):
                alreadyTaken = True

                while alreadyTaken:
                    start = random.randrange(segmentStartRange)
                    end = start + segmentLength

                    alreadyTaken = False

                    for timePair in existingRanges:
                        if (start < timePair[1]) and (timePair[0] < end):
                            alreadyTaken = True

                existingRanges.append([start, end])

                segment = audio[start:end]

                outputPath = outputDir + name + "_" + str(round(start/1000, 2)) + "-" + str(round(end/1000, 2))

                print(outputPath)

                if True: # Prevent accidentally overwriting
                    segment.export(outputPath + ".wav", format="wav")

def voiceActivity():
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

def syllable():
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

def filledPauses():
    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    transcript = []

    totalNumberOfFilledPauses = 0
    totalNumberOfCorrectlyDetectedPauses = 0
    totalNumberOfFalseAlarms = 0

    with open("../media/filled_pauses_validation_participant_audio" + "/filled_pauses.txt") as transcriptFile:
        lines = transcriptFile.readlines()
        for row in lines:
            transcript.append(row.strip().split(', '))

    for line in transcript:
        name = line[0]

        if name[0] != "#":
            actualFilledPausesCount = int(line[1])

            path = None

            # for filePath in sorted(glob.iglob("../media/filled_pauses_validation_participant_audio/" + "*.wav")):
            #     fileName = os.path.basename(filePath)[:-4]
            #
            #     if fileName == name:
            #         path = filePath

            for filePath in sorted(glob.iglob(audioDirectory + "*.wav")):
                fileName = os.path.basename(filePath)[:-4]

                if fileName == name:
                    path = filePath

            if path:
                audio = audioModule.Audio(filePath=path)
                if audio.numberOfChannels != 1:
                    audio.makeMono()

                filledPauses, timeStamps = speechAnalyzer.getFilledPausesFromAudio(audio)

                if True:
                    voiceActivity = speechAnalyzer.getVoiceActivityFromAudio(audio)
                    bufferFrames = int(speechAnalyzer.voiceActivityMaskBufferSize / speechAnalyzer.featureStepSize)
                    mask = np.invert(featureModule.createBufferedBinaryArrayFromArray(voiceActivity.astype(bool), bufferFrames))
                    filledPauses[mask] = 0

                filledPausesMarkers = np.full(int(sum(filledPauses)), 0)
                filledPausesCount = int(sum(filledPauses))

                print(name, "\t", actualFilledPausesCount, filledPausesCount, timeStamps)

                totalNumberOfFilledPauses += actualFilledPausesCount

                if filledPausesCount > actualFilledPausesCount:
                    totalNumberOfFalseAlarms += filledPausesCount - actualFilledPausesCount
                    totalNumberOfCorrectlyDetectedPauses += actualFilledPausesCount
                else:
                    totalNumberOfCorrectlyDetectedPauses += filledPausesCount

    precision = totalNumberOfCorrectlyDetectedPauses / (totalNumberOfCorrectlyDetectedPauses + totalNumberOfFalseAlarms)
    recall = totalNumberOfCorrectlyDetectedPauses / totalNumberOfFilledPauses

    fMeasure = 2 * precision * recall / (precision + recall)

    print("    Total     | Filled pauses:", totalNumberOfFilledPauses)
    print("     New      | Correct filled pauses:", totalNumberOfCorrectlyDetectedPauses,
          "False alarms:", totalNumberOfFalseAlarms, "Precision:", precision,
          "Recall:", recall, "F1", fMeasure)

def main():
    filledPauses()

main()
