from speechLibrary import speechAnalysis
from speechLibrary import audioModule
import numpy

analyzer = speechAnalysis.SpeechAnalyzer()

#Leave default parameters

analyzer.printStatus = True

analyzer.createFeatureFilesFromDirectory("../media/Participant_Audio/", "./featuresCurrent/")
