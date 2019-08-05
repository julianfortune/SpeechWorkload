from speechLibrary import speechAnalysis
from speechLibrary import audioModule
import numpy

analyzer = speechAnalysis.SpeechAnalyzer()

#Leave default parameters

analyzer.printStatus = True
analyzer.lookBackSize = 5

analyzer.createFeatureFilesFromDirectory("../media/Participant_Audio/", "./features/current5second/")
