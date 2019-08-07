from speechLibrary import speechAnalysis
from speechLibrary import audioModule
import numpy

audioDirectory = "../media/Jamison_Evaluations/Supervisory/Day2/"
outputDirectory = "./training/Supervisory_Evaluation_Day_2/features/current5seconds/"

analyzer = speechAnalysis.SpeechAnalyzer()
analyzer.printStatus = True
analyzer.lookBackSize = 5

analyzer.createFeatureFilesFromDirectory(audioDirectory, outputDirectory)
