from speechLibrary import speechAnalysis
from speechLibrary import audioModule
import numpy

import matplotlib.pyplot as plt # Visualisation

oldFeaturesPath = "./features/p1_ol.npy"
oldFeatures = numpy.load(oldFeaturesPath)
numpy.savetxt("oldFeatures.csv", oldFeatures, delimiter=",")

newFeaturesPath = "./test_features/p1_ol.npy"
newFeatures = numpy.load(newFeaturesPath)
numpy.savetxt("newFeatures.csv", newFeatures, delimiter=",")

if numpy.array_equal(oldFeatures, newFeatures):
    print("The two arrays match!")
else:
    print("The two arrays are different.")
    print("There are", newFeatures.size - (oldFeatures == newFeatures).sum(), "changes out of", oldFeatures.size, "entries.")
