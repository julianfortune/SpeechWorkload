from speechLibrary import testModule
import numpy, os, glob
import matplotlib.pyplot as plt # Visualisation

testDir = "./featuresTest/*.npy"
oldDir =  "./features/"

for testPath in sorted(glob.iglob(testDir)):
    oldPath = oldDir + testPath[15:]

    testModule.compareArrayFiles(testPath, oldPath)

    newFeatures = numpy.load(testPath)
    oldFeatures = numpy.load(oldPath)

    seconds = newFeatures[0]

    name = os.path.basename(oldPath)[:-4]

    plt.figure(figsize=[16, 10])
    plt.suptitle(name)

    plt.subplot(711)
    plt.plot(seconds, oldFeatures[1] * 60)
    plt.plot(seconds, newFeatures[1] * 60)
    plt.title("Words Per Minute")
    plt.legend(["old", "new"])

    plt.subplot(712)
    plt.plot(seconds, oldFeatures[2])
    plt.plot(seconds, newFeatures[2])
    plt.title("Voice Activity")

    plt.subplot(713)
    plt.plot(seconds, oldFeatures[3])
    plt.plot(seconds, newFeatures[3])
    plt.title("Voice Activity Standard Deviation")

    plt.subplot(714)
    plt.plot(seconds, oldFeatures[4])
    plt.plot(seconds, newFeatures[4])
    plt.title("Pitch")

    plt.subplot(715)
    plt.plot(seconds, oldFeatures[5])
    plt.plot(seconds, newFeatures[5])
    plt.title("Pitch Standard Deviation")

    plt.subplot(716)
    plt.plot(seconds, oldFeatures[6])
    plt.plot(seconds, newFeatures[6])
    plt.title("Intensity")

    plt.subplot(717)
    plt.plot(seconds, oldFeatures[7])
    plt.plot(seconds, newFeatures[7])
    plt.title("Intensity Standard Deviation")

    plt.subplots_adjust(hspace = 1)
    plt.savefig("./featuresTest/" + name + ".png")

    # plt.show()

    plt.close()
