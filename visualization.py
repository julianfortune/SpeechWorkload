import numpy, os, glob
import matplotlib.pyplot as plt # Visualisation

numpy.set_printoptions(threshold=numpy.inf)

dir = "./features/*.npy"

for path in sorted(glob.iglob(dir)):
    features = numpy.load(path)

    seconds = features[0]
    wpm = features[1]
    va = features[2]
    pitch = features[4]
    intensity = features[6]

    name = os.path.basename(path)[:-4]

    plt.figure()
    plt.suptitle(name)

    plt.subplot(221)
    plt.plot(seconds, wpm)
    plt.title("Words Per Minute")

    plt.subplot(222)
    plt.plot(seconds, va)
    plt.title("Voice Activity")

    plt.subplot(223)
    plt.plot(seconds, pitch)
    plt.title("Pitch")

    plt.subplot(224)
    plt.plot(seconds, intensity)
    plt.title("Intensity")

    # Adjust the subplot layout, because the logit one may take more space
    # than usual, due to y-tick labels like "1 - 10^{-3}"
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)

    plt.savefig("./figures/" + name + ".png")
    plt.close()
