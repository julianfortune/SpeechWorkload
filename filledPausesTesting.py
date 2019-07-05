import numpy as np
import matplotlib.pyplot as plt

import sys, time, glob, os

import wavio
import csv

from pydub import AudioSegment

from speechLibrary import featureModule, speechAnalysis, audioModule

np.set_printoptions(threshold=sys.maxsize)

audioDirectory = "../media/Participant_Audio_First_five/*.wav"
outputDir = "./filledPauses/"

def createSlicesFromPauses():

    for filePath in sorted(glob.iglob(audioDirectory)):

        # Audio file i/o
        name = os.path.basename(filePath)[:-4]

        participant = name.split("_")[0]
        condition = name.split("_")[1]

        # # Make fresh directories
        # os.mkdir(outputDir + name)

        print(participant, condition)

        audio = audioModule.Audio(filePath=filePath)
        audio.makeMono()

        filledPauses = featureModule.getFilledPauses(audio.data, audio.sampleRate, utteranceWindowSize, utteranceStepSize, utteranceMinimumLength, utteranceF1MaximumVariance, utteranceF2MaximumVariance, utteranceEnergyThreshold)

        audio = AudioSegment.from_wav(filePath)

        for time in filledPauses:
            ### Output files - pydub is in ms
            outputPath = outputDir + name + "/" + str(round(time, 2))

            # move back 100 ms
            start = (time - 0.1) * 1000
            # grab a second
            end = (time + 1) * 1000
            segment = audio[start:end]

            # write to disk
            segment.export(outputPath + ".wav", format="wav")

            # move back a whole second for more context
            start = (time - 1) * 1000
            segment = audio[start:end]

            # write to disk
            segment.export(outputPath + "[extra].wav", format="wav")

        # --

        print("Done with ", name)
        print(len(filledPauses))
    # --

# --

def runAlgorithmOnParticipants():

    underLoadFilledPauses = 0
    normalLoadFilledPauses = 0
    overLoadFilledPauses = 0

    participantCount = 30
    directory = "../media/Participant_Audio/"

    filledPausesForParticipant = [["participant","ul","nl","ol"]]

    for participantNumber in range(1, participantCount + 1):
        participantData = [participantNumber]

        for condition in ["ul","nl","ol"]:
            filePath = directory + "p" + str(participantNumber) + "_" + condition + ".wav"

            if filePath != "../media/Participant_Audio/p8_nl.wav":
                print(filePath)

                audio = audioModule.Audio(filePath=filePath)
                audio.makeMono()

                filledPauses = featureModule.getFilledPauses(audio.data, audio.sampleRate, utteranceWindowSize, utteranceStepSize, utteranceMinimumLength, utteranceF1MaximumVariance, utteranceF2MaximumVariance, utteranceEnergyThreshold)

                participantData.append(len(filledPauses))

                print("   ", len(filledPauses))

        print(participantData)
        filledPausesForParticipant.append(participantData)
        print(filledPausesForParticipant)

    with open('./filledPauses/filledPausesForParticipant.csv', 'w') as outputFile:
        writer = csv.writer(outputFile)
        for row in filledPausesForParticipant:
            writer.writerow(row)


def getFeaturesFromSlices():
    filePaths = sorted(glob.iglob("./filledPauses/p3_ol/*extra].wav"))

    analyzer = speechAnalysis.SpeechAnalyzer()

    for filePath in filePaths:
        print(filePath)

        audio = audioModule.Audio(filePath=filePath)
        audio.makeMono()

        filledPauses = analyzer.getFilledPausesFromAudio(audio)

        print(filledPauses)

def getFeaturesFromFile():
    filePath = "../media/cchp_english/p102/p102_en_pd.wav"

    audio = audioModule.Audio(filePath=filePath)
    audio.makeMono()

    print(filePath)

    analyzer = speechAnalysis.SpeechAnalyzer()
    filledPauses = analyzer.getFilledPausesFromAudio(audio)

    print(len(filledPauses))


def runAlgorithmOnSlices():
    analyzer = speechAnalysis.SpeechAnalyzer()

    for subdir, dirs, files in os.walk(outputDir):
        for file in files:
            filePath = os.path.join(subdir, file)

            if "[extra].wav" in filePath:
                print(filePath)
                name = os.path.basename(filePath)[:-4]

                audio = audioModule.Audio(filePath=filePath)
                audio.makeMono()

                filledPauses = analyzer.getFilledPausesFromAudio(audio)

                print(timeStamps)

    # --
# --

def compareAlgorithmToSlices():
    printParameters()
    print("Running on slices")

    analyzer = speechAnalysis.SpeechAnalyzer()

    controlYeses = 0
    controlNos = 0

    yeses = 0
    nos = 0

    startTime = time.time()

    # Compare with file of all existing
    with open('./filledPauses/filledPausesAllParticipantsRatings.csv') as csvfile:
        reader = csv.DictReader(csvfile)

        # Go through each existing filled pause
        for row in reader:
            participant = row['participant']
            condition = row['condition']
            timeStamp = row['time']
            judgement = row['judgement']

            if timeStamp == "862":
                timeStamp = "862.0"

            # Keep track of manual classification
            if judgement == "1":
                controlYeses += 1
            elif judgement == "-1":
                controlNos += 1

            filePath = "./filledPauses/" + participant + "_" + condition[1:] + "/" + timeStamp + "[extra].wav"
            # print(filePath)

            audio = audioModule.Audio(filePath=filePath)
            audio.makeMono()

            # Run algorithm
            filledPauses = analyzer.getFilledPausesFromAudio(audio)

            found = False

            for timeDetected in filledPauses:
                if abs(timeDetected - 1.0) < 0.2 and not found:
                    found = True
                    if judgement == "1":
                        yeses += 1
                    elif judgement == "-1":
                        nos += 1

    print()
    print("  Time to run:", time.time() - startTime)
    print("  Detections:", (yeses + nos), "Accurate detections:", yeses, "Total filled pauses:", controlYeses)
    print("  Precision:", yeses / (yeses + nos))
    print("  Recall:", yeses / controlYeses)
    print("  Score: ", (yeses / controlYeses) * (yeses / (yeses + nos)))
    print()

def testChangesByVaryingParameters():
    testingChanges = [x for x in range(-100, 60, 20)]

    for i in range(0, len(testingChanges)):
        # Hacky
        global utteranceMinimumLength
        utteranceMinimumLength = 200 + testingChanges[i]

        runAlgorithmOnDataset()

def printParameters():
    print()
    print("  utteranceWindowSize:", utteranceWindowSize)
    print("  utteranceStepSize:", utteranceStepSize)
    print("  utteranceMinimumLength:", utteranceMinimumLength)
    print("  utteranceF1MaximumVariance:", utteranceF1MaximumVariance)
    print("  utteranceF2MaximumVariance:", utteranceF2MaximumVariance)
    print("  utteranceEnergyThreshold:", utteranceEnergyThreshold)
    print()

def createMetaDataForDataset():

    train = []
    testing = []
    dev = []
    header = []

    # Load the dataset info for training
    with open('./dr_smart_audio/train/train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        train.extend(reader)

    # Remove and store the header row
    header = train[0]
    train = train[1:]

    # Load the dataset info for training
    with open('./dr_smart_audio/testing/testing.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        testing.extend(reader)

    # Remove the header row
    testing = testing[1:]

    # Load the dataset info for training
    with open('./dr_smart_audio/dev/dev.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        dev.extend(reader)

    # Remove the header row
    dev = dev[1:]

    # Combine datasets
    dataset = train + testing + dev

    # Change the file paths and string to count
    for row in dataset:
        row[0] = row[0][38:]
        row[1] = row[2].count("um") + row[2].count("uh")
        row.pop(2)

    # Reinsert the header row
    header[1] = "filled_pause_count"
    header.pop(2)
    dataset.insert(0, header)


    with open('./dr_smart_audio/metadata.csv', 'w') as outputFile:
        writer = csv.writer(outputFile)
        for row in dataset:
            writer.writerow(row)

def compareAlgorithmToDataset():
    printParameters()
    print("Running on Dr. Smart's Dataset")

    analyzer = speechAnalysis.SpeechAnalyzer()

    directory = './dr_smart_audio'
    dataset = []

    numberOfAccurateDetections = 0
    numberOfDetections = 0
    trueNumberOfFilledPauses = 0

    # Load the dataset info for training
    with open(directory + '/metadata.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        dataset.extend(reader)

    # Remove header
    dataset.pop(0)

    startTime = time.time()

    for audioFile in dataset:
        audio = audioModule.Audio(filePath=directory + audioFile[0])

        filledPauses = analyzer.getFilledPausesFromAudio(audio)

        if int(audioFile[1]) <= len(filledPauses):
            numberOfAccurateDetections += int(audioFile[1])

        trueNumberOfFilledPauses += int(audioFile[1])
        numberOfDetections += len(filledPauses)

    print()
    print("  Time to run:", time.time() - startTime)
    print("  Detections:", numberOfDetections, "Accurate detections:", numberOfAccurateDetections, "Total filled pauses:", trueNumberOfFilledPauses)
    print("  Precision:", numberOfAccurateDetections / numberOfDetections)
    print("  Recall:", numberOfAccurateDetections / trueNumberOfFilledPauses)
    print("  Score: ", (numberOfAccurateDetections / numberOfDetections) * (numberOfAccurateDetections / trueNumberOfFilledPauses))
    print()

def runAlgorithmOnDataset():
    directory = './dr_smart_audio'
    dataset = []

    analyzer = speechAnalysis.SpeechAnalyzer()

    # Load the dataset info for training
    with open(directory + '/metadata.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        dataset.extend(reader)

    # Remove header
    dataset.pop(0)

    for audioFile in dataset:
        filePath = audioFile[0]

        print(filePath)

        audio = audioModule.Audio(filePath=directory + audioFile[0])

        filledPauses = analyzer.getFilledPausesFromAudio(audio)
        print(len(filledPauses))


def main():
    compareAlgorithmToDataset()

main()
