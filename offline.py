from speechLibrary import speechAnalysis
from speechLibrary import audioModule
import numpy

analyzer = speechAnalysis.SpeechAnalyzer()

analyzer.printStatus = True

featureArray = analyzer.getFeaturesFromFile("../media/Participant_Audio/p1_ol.wav")

numpy.save("./test_features/p1_ol", featureArray)
