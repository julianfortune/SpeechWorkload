import numpy as np
import matplotlib.pyplot as plt

import sys, time, glob, os

import wavio
import csv

from pydub import AudioSegment

from speechLibrary import featureModule, speechAnalysis, audioModule

np.set_printoptions(threshold=sys.maxsize)

# Parameters of the features
utteranceWindowSize = 30 # milliseconds
utteranceStepSize = 15 # milliseconds
utteranceMinimumLength = 200 # milliseconds
utteranceF1MaximumVariance = 40
utteranceF2MaximumVariance = 40
utteranceEnergyThreshold = 60

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

        filledPauses, timeStamps, times, f1, f2, energy, lengths, firstFormantVariances, secondFormantVariances, averageEnergies, stepTimes = featureModule.getFilledPauses(audio.data, audio.sampleRate, utteranceWindowSize, utteranceStepSize, utteranceMinimumLength, utteranceF1MaximumVariance, utteranceF2MaximumVariance, utteranceEnergyThreshold)

        audio = AudioSegment.from_wav(filePath)

        for time in timeStamps:
            ### Output files — pydub is in ms
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
        print(sum(filledPauses))
    # --

# --

def compareAlgorithmToSlices():

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

        filledPauses, timeStamps, times, f1, f2, energy, lengths, firstFormantVariances, secondFormantVariances, averageEnergies, stepTimes = featureModule.getFilledPauses(audio.data, audio.sampleRate, utteranceWindowSize, utteranceStepSize, utteranceMinimumLength, utteranceF1MaximumVariance, utteranceF2MaximumVariance, utteranceEnergyThreshold)

        audio = AudioSegment.from_wav(filePath)

        for time in timeStamps:

            # Compare with file containing marked pauses
            with open('filledPausesAllParticipantsRatings.csv') as csvfile:
                reader = csv.DictReader(csvfile)

                # Go through each existing filled pause
                for row in reader:
                    controlParticipant = row['participant']
                    controlCondition = row['condition']
                    controlTime = row['time']
                    judgement = row['judgement']

                    if controlParticipant == participant and controlCondition[1:] == condition:
                        if abs(time - float(controlTime)) < 0.01:
                            if judgement == "1":
                                print("correct")
                            elif judgement == "0":
                                print("maybe")
                            elif judgement == "-1":
                                print("wrong")


def runAlgorithmOnParticipants():

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

        filledPauses, timeStamps, times, f1, f2, energy, lengths, firstFormantVariances, secondFormantVariances, averageEnergies, stepTimes = featureModule.getFilledPauses(audio.data, audio.sampleRate, utteranceWindowSize, utteranceStepSize, utteranceMinimumLength, utteranceF1MaximumVariance, utteranceF2MaximumVariance, utteranceEnergyThreshold)

        audio = AudioSegment.from_wav(filePath)

        for time in timeStamps:

            # go back a second from filled pause
            end = (time + 1) * 1000
            # move forward a second
            start = (time - 1) * 1000

            # Graphing
            start = int(start/utteranceStepSize)
            end = int(end/utteranceStepSize)

            filledPausesMarkers = [1] * len(timeStamps)
            energyThresholdMarkers = [utteranceEnergyThreshold] * len(times)
            firstFormatVarianceMarkers = [utteranceF1MaximumVariance] * len(stepTimes)
            secondFormatVarianceMarkers = [utteranceF2MaximumVariance] * len(stepTimes)

            fig, axs = plt.subplots(4, 1,figsize=(7,7))
            axs[0].plot(times[start:end], f1[start:end], times[start:end], f2[start:end], times[start:end], energy[start:end], time, 1, 'ro')
            axs[0].set_title('Formants and Energy')
            axs[1].plot(stepTimes[start:end], firstFormantVariances[start:end], stepTimes[start:end], firstFormatVarianceMarkers[start:end], time, 1, 'ro')
            axs[1].set_title('First Formant Variance')
            axs[2].plot(stepTimes[start:end], secondFormantVariances[start:end], stepTimes[start:end], secondFormatVarianceMarkers[start:end], time, 1, 'ro')
            axs[2].set_title('Second Formant Variance')
            axs[3].plot(times[start:end], energy[start:end], times[start:end], energyThresholdMarkers[start:end], time, 1, 'ro')

            fig.tight_layout()

            plt.savefig(outputPath + "[extra].png")
            plt.close()

def runAlgorithmOnSlices():

    for subdir, dirs, files in os.walk(outputDir):
        for file in files:
            filePath = os.path.join(subdir, file)

            if "[extra].wav" in filePath:
                print(filePath)
                name = os.path.basename(filePath)[:-4]

                audio = audioModule.Audio(filePath=filePath)
                audio.makeMono()

                filledPauses, timeStamps, times, f1, f2, energy, lengths, firstFormantVariances, secondFormantVariances, averageEnergies, stepTimes = featureModule.getFilledPauses(audio.data, audio.sampleRate, utteranceWindowSize, utteranceStepSize, utteranceMinimumLength, utteranceF1MaximumVariance, utteranceF2MaximumVariance, utteranceEnergyThreshold)

                print(timeStamps)

                filledPausesMarkers = [1] * len(timeStamps)
                energyThresholdMarkers = [utteranceEnergyThreshold] * len(times)
                firstFormatVarianceMarkers = [utteranceF1MaximumVariance] * len(stepTimes)
                secondFormatVarianceMarkers = [utteranceF2MaximumVariance] * len(stepTimes)

                fig, axs = plt.subplots(4, 1)
                axs[0].plot(times, f1, times, f2, times, energy, timeStamps, filledPausesMarkers, 'ro')
                axs[0].set_title('Formants and Energy')
                axs[1].plot(stepTimes, firstFormantVariances, stepTimes, firstFormatVarianceMarkers, timeStamps, filledPausesMarkers, 'ro')
                axs[1].set_title('First Formant Variance')
                axs[2].plot(stepTimes, secondFormantVariances, stepTimes, secondFormatVarianceMarkers, timeStamps, filledPausesMarkers, 'ro')
                axs[2].set_title('Second Formant Variance')
                axs[3].plot(times, energy, times, energyThresholdMarkers, timeStamps, filledPausesMarkers, 'ro')

                fig.tight_layout()

                # plt.savefig(outputPath + "[extra].png")
                plt.show()

    # --
# --

def checkNewAlgorithmAgainstSlices():

    yeses = 0
    maybes = 0
    nos = 0

    for filePath in sorted(glob.iglob(audioDirectory)):
        # Audio file i/o
        name = os.path.basename(filePath)[:-4]

        participant = name.split("_")[0]
        condition = name.split("_")[1]

        print(participant, condition)

        audio = audioModule.Audio(filePath=filePath)
        audio.makeMono()

        # Run algorithm
        filledPauses, timeStamps, times, f1, f2, energy, lengths, firstFormantVariances, secondFormantVariances, averageEnergies, stepTimes = featureModule.getFilledPauses(audio.data, audio.sampleRate, utteranceWindowSize, utteranceStepSize, utteranceMinimumLength, utteranceF1MaximumVariance, utteranceF2MaximumVariance, utteranceEnergyThreshold)

        # Compare with file of all existing
        with open('filledPausesAllParticipantsRatings.csv') as csvfile:
            reader = csv.DictReader(csvfile)

            # Go through each existing filled pause
            for row in reader:
                controlParticipant = row['participant']
                controlCondition = row['condition']
                controlTime = row['time']
                judgement = row['judgement']

                if controlParticipant == participant and controlCondition[1:] == condition:
                    name = participant + "_" + condition[1:]

                    found = False

                    for time in timeStamps:
                        if abs(time - float(controlTime)) < 0.01:
                            if judgement == "1":
                                yeses += 1
                            elif judgement == "0":
                                maybes += 1
                            elif judgement == "-1":
                                nos += 1

    # Print original accuracy
    with open('filledPausesAllParticipantsRatings.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        controlYeses = 0
        controlMaybes = 0
        controlNos = 0

        # Check accuracy of original
        for row in reader:
            judgement = row['judgement']

            if judgement == "1":
                controlYeses += 1
            elif judgement == "0":
                controlMaybes += 1
            elif judgement == "-1":
                controlNos += 1
        print()
        print("  Original yeses:", controlYeses, "maybes:", controlMaybes, "nos:", controlNos)

    print("  Yeses:", yeses, "maybes:", maybes, "nos:", nos)
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

def runAlgorithmOnDataset():
    printParameters()

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

        filledPauses, timeStamps, times, f1, f2, energy, lengths, firstFormantVariances, secondFormantVariances, averageEnergies, stepTimes = featureModule.getFilledPauses(audio.data, audio.sampleRate, utteranceWindowSize, utteranceStepSize, utteranceMinimumLength, utteranceF1MaximumVariance, utteranceF2MaximumVariance, utteranceEnergyThreshold)

        if int(audioFile[1]) <= len(timeStamps):
            numberOfAccurateDetections += int(audioFile[1])

        trueNumberOfFilledPauses += int(audioFile[1])
        numberOfDetections += len(timeStamps)

        # Graphing
        filledPausesMarkers = [1] * len(timeStamps)
        energyThresholdMarkers = [utteranceEnergyThreshold] * len(stepTimes)
        firstFormatVarianceMarkers = [utteranceF1MaximumVariance] * len(stepTimes)
        secondFormatVarianceMarkers = [utteranceF2MaximumVariance] * len(stepTimes)

        fig, axs = plt.subplots(4, 1)
        axs[0].plot(times, f1, times, f2, timeStamps, filledPausesMarkers, 'ro')
        axs[0].set_title('Formants and Energy')
        axs[1].plot(stepTimes, firstFormantVariances, stepTimes, firstFormatVarianceMarkers, timeStamps, filledPausesMarkers, 'ro')
        axs[1].set_title('First Formant Variance')
        axs[2].plot(stepTimes, secondFormantVariances, stepTimes, secondFormatVarianceMarkers, timeStamps, filledPausesMarkers, 'ro')
        axs[2].set_title('Second Formant Variance')
        axs[3].plot(times, energy, timeStamps, filledPausesMarkers, 'ro')
        axs[3].set_title('Intensity')

        fig.suptitle(audioFile[0])
        fig.tight_layout()
        plt.show()

        print(audioFile[0], timeStamps, lengths)

    print("Time to run:", time.time() - startTime)
    print("Detections:", numberOfDetections, "Accurate detections:", numberOfAccurateDetections, "Total filled pauses:", trueNumberOfFilledPauses)
    print("Precision:", numberOfAccurateDetections / numberOfDetections)
    print("Recall:", numberOfAccurateDetections / trueNumberOfFilledPauses)
    print("Score: ", (numberOfAccurateDetections / numberOfDetections) * (numberOfAccurateDetections / trueNumberOfFilledPauses))


def main():
    printParameters()
    checkNewAlgorithmAgainstSlices()

main()
