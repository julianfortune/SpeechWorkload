from speechLibrary import speechAnalysis
from speechLibrary import audioModule
import numpy

def createFeatures(audioDirectory, outputDirectory):
    analyzer = speechAnalysis.SpeechAnalyzer()

    analyzer.printStatus = True
    analyzer.lookBackSize = 5

    analyzer.createFeatureFilesFromDirectory(audioDirectory, outputDirectory)

def createRealTimeFeaturesWithVaryingWindowSize():
    audioDirectory= "../media/Jamison_Evaluations/Real_Time_Evaluation/Audio/"
    outputDirectory= "./training/Real_Time"

    windowSizes = [1, 5, 10, 15, 30, 60]

    analyzer = speechAnalysis.SpeechAnalyzer()
    analyzer.printStatus = True

    for windowSize in windowSizes:
        analyzer.lookBackSize = windowSize
        print(outputDirectory + "-" + str(windowSize) + "_second_window/features/", analyzer.lookBackSize)

        analyzer.createFeatureFilesFromDirectory(audioDirectory, outputDirectory + "-" + str(windowSize) + "_second_window/features/", saveRunTimes= True)

def day1():
    createFeatures(audioDirectory= "../media/Jamison_Evaluations/Supervisory/Day1/",
                   outputDirectory= "./training/Supervisory_Evaluation_Day_1/features/")

def day2():
    createFeatures(audioDirectory= "../media/Jamison_Evaluations/Supervisory/Day2/",
                   outputDirectory= "./training/Supervisory_Evaluation_Day_2/features/")

def peerBased():
    createFeatures(audioDirectory= "../media/Jamison_Evaluations/Peer_Based/Processed_Audio/",
                   outputDirectory= "./training/Peer_Based/features/")

def realTime():
    createFeatures(audioDirectory= "../media/Jamison_Evaluations/Real_Time_Evaluation/Audio/",
                   outputDirectory= "./training/Real_Time/features/")

def main():
    createRealTimeFeaturesWithVaryingWindowSize()

main()
