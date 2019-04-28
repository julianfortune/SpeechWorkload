'''
Created on Apr 23, 2019

@author: Julian Fortune

@Description: Interface for extracting feature arrays.
'''
import os, glob # file io
import time
import numpy
import pyaudio # microphone io
import wavio # microphone decoding
from speechLibrary import featureModule
from speechLibrary import audioModule

numpy.set_printoptions(threshold=numpy.nan)

### Utilities

# | A class used to perfom the same analysis on any number of files or on
# | live input. Handles file and microphone IO in order to manage the entire
# | process.
class SpeechAnalyzer:

    def __init__(self):

        # Set defaults for all parameters
        self.printStatus = True

        # Windowing parameters
        self.stepSize = 1 #how big to step in seconds
        self.lookBackSize = 30  #how big of interval to wait until looking for transcript, pitch/intensity features in seconds

        # Syllable detection parameters
        self.syllableWindowSize = 64 # In milliseconds
        self.syllableStepSize = 16 # In milliseconds
        self.syllablePeakMinDistance = 5
        self.syllablePeakMinWidth = 2
        self.syllableZcrThreshold = 0.06
        self.syllableDominantFreqThreshold = 200
        self.syllableDominantFreqTolerance = 8

        # Voice activity Parameters
        self.voiceActivityIsAdaptive = True
        # From "A Simple but efficient..."
        self.voiceActivityWindowSize = 10 # In milliseconds
        self.voiceActivityStepSize = 10 # In milliseconds
        self.voiceActivityZCRThreshold = 0.06
        self.voiceActivityEnergyThreshold = 40
        self.voiceActivityFreqThreshold = 18
        self.voiceActivityFreqTolerance = 8

        # Recording parameters
        self.recordingDeviceIndex = -1
        self.recordingBufferSize = 4096
        self.recordingFormat = pyaudio.paInt16
        self.recordingChannels = 2

    def getFeaturesFromAudio(self, audio):

        features = featureModule.FeatureSet()

        ### WORDS PER MINUTE
        syllables = featureModule.getSyllables(data=audio.data,
                                               sampleRate=audio.sampleRate,
                                               windowSize=self.syllableWindowSize,
                                               stepSize=self.syllableStepSize,
                                               peakMinDistance=self.syllablePeakMinDistance,
                                               peakMinWidth=self.syllablePeakMinWidth,
                                               zcrThreshold=self.syllableZcrThreshold,
                                               dominantFreqThreshold=self.syllableDominantFreqThreshold,
                                               dominantFreqTolerance=self.syllableDominantFreqTolerance)

        currentSyllablesPerSecond = len(syllables)/self.lookBackSize
        features.syllablesPerSecond = numpy.append(features.syllablesPerSecond, currentSyllablesPerSecond)

        ### VAD
        average, stDev = featureModule.getVoiceActivityFeatures(data=audio.data,
                                                                sampleRate=audio.sampleRate,
                                                                windowSizeInMS=self.voiceActivityWindowSize,
                                                                stepSizeInMS=self.voiceActivityStepSize,
                                                                useAdaptiveThresholds=self.voiceActivityIsAdaptive,
                                                                zcrThreshold=self.voiceActivityZCRThreshold,
                                                                energyPrimaryThreshold=self.voiceActivityEnergyThreshold,
                                                                dominantFreqThreshold=self.voiceActivityFreqThreshold,
                                                                dominantFreqTolerance=self.voiceActivityFreqTolerance)

        features.meanVoiceActivity = numpy.append(features.meanVoiceActivity,average)
        features.stDevVoiceActivity = numpy.append(features.stDevVoiceActivity,stDev)

        ### AVERAGE PITCH
        average, stDev = featureModule.getPitchFeatures(audio.data,
                                                        audio.sampleRate,
                                                        1)
        features.meanPitch = numpy.append(features.meanPitch, average)
        features.stDevPitch = numpy.append(features.stDevPitch, stDev)

        ### INTENSITY FEATURES
        average, stDev = featureModule.getIntensityFeatures(audio.data)
        features.meanIntensity = numpy.append(features.meanIntensity, average)
        features.stDevIntensity = numpy.append(features.stDevIntensity, stDev)

        return features

    # | Extracts all features and returns array in accordance with Jamison's drawing
    # | Parameters:
    # |   - filePath: path to file to process
    # | Returns:
    # |   - Numpy array with features
    def getFeaturesFromFile(self, filePath):

        if self.printStatus :
            print("[ START ] Working on:",filePath)

        # Read in the file, extract data and metadata
        audio = audioModule.Audio(filePath)

        if audio.numberOfChannels > 1:
            audio.makeMono()

        # Set up time tracker
        seconds = numpy.zeros(shape=0)

        # Set up arrays for features
        features = featureModule.FeatureSet()

        step = 0
        sampleStepSize = int(self.stepSize * audio.sampleRate)
        sampleLookBackSize = int(self.lookBackSize * audio.sampleRate)

        while step < audio.length:

            if self.printStatus:
                print("[",str(step/audio.length*100)[:4],"% ] Second",int(step/audio.sampleRate), end="\r")

            # Keep track of what second we're in
            seconds = numpy.append(seconds,step/audio.sampleRate)

            # Look backward to calculate features over long term
            if step + sampleStepSize - sampleLookBackSize > 0:

                currentWindow = audioModule.Audio(data=audio.data[step + sampleStepSize - sampleLookBackSize:step + sampleStepSize])
                currentWindow.sampleRate = audio.sampleRate

                currentFeatures = self.getFeaturesFromAudio(currentWindow)

                features.append(currentFeatures)

            # Fills arrays with zeros until step is larger than lookBackSize
            else:
                features.appendAllZeros()

            # Increment to next step
            step += sampleStepSize

        # Pulls all the feautures together in one array
        featureArray = numpy.vstack([seconds,
                                     features.syllablesPerSecond,
                                     features.meanVoiceActivity,
                                     features.stDevVoiceActivity,
                                     features.meanPitch,
                                     features.stDevPitch,
                                     features.meanIntensity,
                                     features.stDevIntensity])

        if self.printStatus :
            print("[ DONE ] Finished processing",filePath,"!")

        return featureArray

    def getFeaturesOnAllFilesInDirectory(self):
        dir = "../media/Participant_Audio/*.wav"

        # Keep track of running stats
        startTime = time.time()
        count = 1

        for path in sorted(glob.iglob(dir),reverse=False):
            # Communicate progress
            print("[ " + str(count) + "/" + str(len(sorted(glob.iglob(dir)))) + " ] \tStarting:",path)

            featureArray = getFeaturesFromFile(path)

            # Save the numpy array
            numpy.save("./test_features/" + os.path.basename(path)[:-4],featureArray)

            # Crunch some numbers and communicate to the user
            timeElapsed = time.time() - startTime
            estimatedTimeRemaining = timeElapsed/count * (len(sorted(glob.iglob(dir))) - count)
            print("\t\t" + str(timeElapsed/60) + " minutes elapsed. Estimated time remaining: " + str(estimatedTimeRemaining/60))

            count += 1

    def getFeaturesFromLiveInput(self):
        audioController = pyaudio.PyAudio()

        microphoneIndex = self.recordingDeviceIndex

        if microphoneIndex < 0 or microphoneIndex >= audioController.get_device_count():
            microphoneIndex = audioModule.pickMicrophone(audioController)

        sampleRate = int(audioController.get_device_info_by_index(microphoneIndex)["defaultSampleRate"])

        input("Press 'Enter' to start recording. Use keyboard interrupt to stop.")

        audioStream = audioController.open(format=self.recordingFormat,
                                           input_device_index=microphoneIndex,
                                           channels=self.recordingChannels,
                                           rate=sampleRate,
                                           input=True,
                                           frames_per_buffer=self.recordingBufferSize)

        print("\n\u001b[31m• Live\u001b[0m\r", end="\r")

        data = numpy.zeros(shape=0)

        windowSampleSize = int(sampleRate * self.lookBackSize)
        stepSampleSize = int(sampleRate * self.stepSize)

        startTime = time.time()

        try:
            # Record until the user stops
            while True:
                # Read in from microphone
                buffer = audioStream.read(self.recordingBufferSize)

                # Convert to mono float data
                waveData = wavio._wav2array(self.recordingChannels, audioController.get_sample_size(self.recordingFormat), buffer)
                monoWaveData = numpy.mean(waveData,axis=1)

                # Add the just-read buffer to the current running array of audio data
                data = numpy.append(data, monoWaveData)

                # Once enough time has passed to include an entire window
                if data.size >= windowSampleSize:

                    # Chop out a window sized section of data
                    currentWindowSelection = data[0:windowSampleSize]

                    # Create an object to pass around
                    audio = audioModule.Audio(data=currentWindowSelection)
                    audio.sampleRate = sampleRate
                    audio.numberOfChannels = 2

                    # Get all features for this window of audio
                    features = self.getFeaturesFromAudio(audio)

                    print("\u001b[31m• Live\u001b[0m", str((time.time() - startTime))[:5], "elapsed.",
                          "Mean pitch:", features.meanPitch[0],
                          "Voice activity:", features.meanVoiceActivity[0],
                          "Speech Rate:", features.syllablesPerSecond[0],
                          end="      \r")

                    # Reduce the size of the audio data array to move the beggining
                    # forward by one step size so the next window is offset by this amount
                    data = data[stepSampleSize:]

        # User stops with ctrl + c
        except KeyboardInterrupt:
            print("\rStopped!                                   ")

        # Clean up audio session
        audioStream.stop_stream()
        audioStream.close()
        audioController.terminate()
