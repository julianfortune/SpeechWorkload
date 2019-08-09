#
# Created on August 2, 2019
#
# @author: Julian Fortune
# @Description: Validation of filled pauses algorithm.
#

import sys, time, glob, os
import numpy as np
import matplotlib.pyplot as plt

from speechLibrary import featureModule, speechAnalysis, audioModule

np.set_printoptions(threshold=sys.maxsize)

def validateWithTranscript():
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

def validateWithSVCCorpus():
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

def testingCCHP():
    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    # audio = audioModule.Audio(filePath="../media/SBC001.wav")
    audio = audioModule.Audio(filePath="../media/cchp_english/p102/p102_en_pd.wav")
    if audio.numberOfChannels != 1:
        audio.makeMono()

    filledPauses, timeStamps = speechAnalyzer.getFilledPausesFromAudio(audio)
    print(timeStamps)

def validateWithCCHP():
    corpusTopLevelPath = "../media/cchp_english/"
    speechAnalyzer = speechAnalysis.SpeechAnalyzer()

    # Iterate through sub directories with participants.
    for participantPath in sorted(glob.iglob(corpusTopLevelPath + '*/')):

        totalNumberOfFilledPauses = 0
        totalNumberOfCorrectlyDetectedPauses = 0
        totalNumberOfFalseAlarms = 0

        # Find the audio files for each condition.
        for filePath in sorted(glob.iglob(participantPath + "*.wav")):
            fileName = os.path.basename(filePath)[:-4]

            # Find the matching transcript
            for transciptPath in sorted(glob.iglob(participantPath + "*.xml")):
                transcriptName = os.path.basename(transciptPath)[:-4]

                if fileName == transcriptName:
                    # Grab the number of filled pauses
                    transcriptFile  = open(transciptPath, 'r').read()
                    actualFilledPausesCount = transcriptFile.count("uh</T>") + transcriptFile.count("um</T>")  +transcriptFile.count("mm</T>")

                    audio = audioModule.Audio(filePath=filePath)
                    if audio.numberOfChannels == 2:
                        audio.makeMono()

                    _, timeStamps = speechAnalyzer.getFilledPausesFromAudio(audio)

                    algorithmFilledPauseCount = len(timeStamps)

                    totalNumberOfFilledPauses += actualFilledPausesCount

                    if algorithmFilledPauseCount > actualFilledPausesCount:
                        totalNumberOfFalseAlarms += algorithmFilledPauseCount - actualFilledPausesCount
                        totalNumberOfCorrectlyDetectedPauses += actualFilledPausesCount
                    else:
                        totalNumberOfCorrectlyDetectedPauses += algorithmFilledPauseCount

                    print(fileName, actualFilledPausesCount, algorithmFilledPauseCount)

        # precision = totalNumberOfCorrectlyDetectedPauses / (totalNumberOfCorrectlyDetectedPauses + totalNumberOfFalseAlarms)
        # recall = totalNumberOfCorrectlyDetectedPauses / totalNumberOfFilledPauses
        #
        # f1 = 2 * precision * recall / (precision + recall)

        print("    Total     | Filled pauses:", totalNumberOfFilledPauses)
        print("     New      | Correct filled pauses:", totalNumberOfCorrectlyDetectedPauses, "False alarms:", totalNumberOfFalseAlarms)
        # "Precision:", precision, "Recall:", recall, "F1", f1)

def main():
    validateWithCCHP()

main()
