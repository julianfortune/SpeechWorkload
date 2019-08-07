from speechLibrary import speechAnalysis
from speechLibrary import audioModule
import numpy

audioDirectory = "../media/Jamison_Evaluation/Supervisory/Day1"
outputDirectory = "./training/Supervisory_Evaluation_Day_1/features/current5second/"

analyzer = speechAnalysis.SpeechAnalyzer()
analyzer.printStatus = True
analyzer.lookBackSize = 5

analyzer.createFeatureFilesFromDirectory(audioDirectory, outputDirectory)
