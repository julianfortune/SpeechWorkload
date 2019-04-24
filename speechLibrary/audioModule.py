'''
Created on Apr 23, 2019

@author: Julian Fortune

@Description: Class for managing audio data and metadata.
'''
import wavio
import wave
import pyaudio
import numpy

class Audio:

    format = pyaudio.paInt16

    # Convenience constructor from audio file or data
    def __init__(self, filePath=None, data=None):
        self.data = numpy.zeros(shape=0)
        self.numberOfChannels = 0
        self.sampleRate = 0
        self.width = 0
        self.length = 0

        if isinstance(filePath, str):
            # Read in the file, extract data and metadata
            audioData = wavio.read(filePath) # reads in audio file
            self.data = audioData.data
            self.numberOfChannels = len(audioData.data[0])
            self.sampleRate = audioData.rate # usually 44100
            self.width = audioData.sampwidth # bit depth is equal to width times 8
            self.length = len(audioData.data) # gets number of sample in audio

        if isinstance(data, numpy.ndarray):
            self.data = data
            self.length = len(data)

    # Save audio in the wav format
    def writeToFile(self, filePath):
        outputFile = wave.open(filePath, 'wb')
        outputFile.setnchannels(self.numberOfChannels)
        outputFile.setsampwidth(audioController.get_sample_size(format))
        outputFile.setframerate(self.sampleRate)
        outputFile.writeframes(b''.join(self.data))
        outputFile.close()

    # Make the audio data mono
    def makeMono(self):
        if self.numberOfChannels > 1:
            self.data = numpy.mean(self.data,axis=1)
            self.numberOfChannels = 1
        else:
            print("ALREADY MONO")

    def description(self):
        print("Audio data - ", self.sampleRate, "samples/second,", self.numberOfChannels, "channels,", self.length, "seconds.")

# Uses the pyaudio library to get the user's choice of input device
def pickMicrophone(pyAudioController):
    devices = pyAudioController.get_device_count()
    print("\u001b[35m" + "Available devices:" + "\u001b[0m")
    for deviceIndex in range(0, devices):
        info = pyAudioController.get_device_info_by_index(deviceIndex)
        print("   ", deviceIndex, "-", info["name"], "\t . . . Max input channels:", info["maxInputChannels"])
    print()
    selectedIndex = int(input("Which input device would you like to use: "))
    print("Using: ", pyAudioController.get_device_info_by_index(selectedIndex)["name"])
    return selectedIndex
