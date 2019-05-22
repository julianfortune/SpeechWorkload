from speechLibrary import speechAnalysis
from speechLibrary import audioModule
import numpy

analyzer = speechAnalysis.SpeechAnalyzer()

analyzer.printStatus = True

featureArray = analyzer.getFeaturesFromFileUsingWindowing("../media/Participant_Audio/p1_nl.wav")

numpy.save("./features/p1_nl", featureArray)
