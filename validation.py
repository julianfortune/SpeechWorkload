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

def syllableUsingTranscript():
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

def syllablesUsingPRAATScript():
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

def syllablesUsingHarvardSentences():
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

def testingWithSilentNonSilentAudio():
    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    praat = []
    algorithm = []

    praatOutput = []

    validationSetPath = "../media/silent_praat_testing/"

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
    plt.xlim(-0.5, 80)
    plt.ylim(-0.5, 40)
    plt.show()

def filledPausesUsingTranscript():
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

def filledPausesWithSVCCorpus():
    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    corpusPath = "../media/vocalizationcorpus"
    labelsPath = corpusPath + "/labels.txt"

    transcript = []

    totalNumberOfFilledPauses = 0
    totalNumberOfCorrectlyDetectedPauses = 0
    totalNumberOfFalseAlarms = 0

    with open(labelsPath) as transcriptFile:
        lines = transcriptFile.readlines()
        for row in lines:
            transcript.append(row.strip().split(','))

    # Remove header line
    transcript.pop(0)

    for row in transcript:
        fileName = row[0]

        utterances = row[4:]

        # print(fileName, utterances)

        utterances = np.array(utterances)
        utterances = utterances.reshape((int(utterances.shape[0]/3)), 3)

        if 'filler' in utterances:
            filePath = corpusPath + "/data/" + fileName + ".wav"

            audio = audioModule.Audio(filePath=filePath)
            if audio.numberOfChannels != 1:
                audio.makeMono()

            filledPauses, timeStamps = speechAnalyzer.getFilledPausesFromAudio(audio)

            for utterance in utterances:
                if utterance[0] == "filler":
                    totalNumberOfFilledPauses += 1

            for filledPauseDetectedTime in timeStamps:
                correctDetection = False
                for utterance in utterances:
                    if utterance[0] == "filler" and abs(float(utterance[1]) - filledPauseDetectedTime) < 0.5:
                        correctDetection = True

                if correctDetection:
                    totalNumberOfCorrectlyDetectedPauses += 1
                else:
                    totalNumberOfFalseAlarms += 1

            print(fileName, totalNumberOfFilledPauses, totalNumberOfCorrectlyDetectedPauses, totalNumberOfFalseAlarms)

def testing():
    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    # audio = audioModule.Audio(filePath="../media/SBC001.wav")
    audio = audioModule.Audio(filePath="../media/cchp_english/p102/p102_en_pd.wav")
    if audio.numberOfChannels != 1:
        audio.makeMono()

    filledPauses, timeStamps = speechAnalyzer.getFilledPausesFromAudio(audio)
    print(timeStamps)

def main():
    # createMultipleValidationSets(participantDirectoryPath= "../media/Participant_Audio/",
    #                              outputDirectoryPath= "../media/validationTesting/",
    #                              numberOfSets= 10,
    #                              segmentLengthInSeconds= 30)

    # filledPausesWithSVCCorpus()

    testingWithSilentNonSilentAudio()

main()
