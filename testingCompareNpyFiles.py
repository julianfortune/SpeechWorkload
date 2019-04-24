from speechLibrary import speechAnalysis
from speechLibrary import audioModule
import numpy

oldFeaturesPath = "./features/p1_ol.npy"
oldFeatures = numpy.load(oldFeaturesPath)
numpy.savetxt("oldFeatures.csv", oldFeatures, delimiter=",")

newFeaturesPath = "./test_features/p1_ol.npy"
newFeatures = numpy.load(newFeaturesPath)
numpy.savetxt("newFeatures.csv", newFeatures, delimiter=",")

print(numpy.array_equal(oldFeatures, newFeatures))
