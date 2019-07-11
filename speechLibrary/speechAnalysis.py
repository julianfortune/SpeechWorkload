#
# Created on Apr 23, 2019
#
# @author: Julian Fortune
# @Description: Interface for extracting feature arrays.
#

import os, glob, sys # file io
import time
import numpy as np

import pyaudio # microphone io
import wavio # microphone decoding

import csv

from speechLibrary import featureModule, audioModule

np.set_printoptions(threshold=sys.maxsize)

DEBUG = False

FEATURE_NAMES = ["meanIntensity", "stDevIntensity", "meanPitch", "stDevPitch", "meanVoiceActivity", "stDevVoiceActivity", "syllablesPerSecond", "filledPauses"]

# | A class used to perfom the same analysis on any number of files or on
# | live input. Handles file and microphone IO in order to manage the entire
# | process.
class SpeechAnalyzer:

    def __init__(self):
        self.printStatus = True

        # Windowing parameters
        self.stepSize = 1 # Time interval between extracting features, in seconds.
        self.lookBackSize = 30 # Time interval to examine looking backward for features, in seconds.

        # Parameters for all features
        self.features = ["syllables", "meanVoiceActivity", "stDevVoiceActivity", "meanPitch", "stDevPitch", "meanIntensity", "stDevIntensity"]
        self.featureStepSize = 10 # Step size for all features, in milliseconds.
        self.energyThresholdRatio = 4

        self.voiceActivityMaskBufferSize = 100 # In milliseconds

        self.maskEnergyWithVoiceActivity = False
        self.maskPitchWIthVoiceActivity = False
        self.maskSyllablesWithVoiceActivity = False
        self.maskFilledPausesWithVoiceActivity = False

        # Pitch parameters
        self.energyWindowSize = 50

        # Pitch parameters
        self.pitchMinimumRunLength = 4

        # Voice activity Parameters
        self.voiceActivityIsAdaptive = True
        self.voiceActivityWindowSize = 10 # In milliseconds
        self.voiceActivityZCRThreshold = 0.04 # Originally 0.06
        self.voiceActivityEnergyThreshold = 40
        self.voiceActivityPitchTolerance = 8
        self.voiceActivityMinimumRunLength = 10

        # Syllable detection parameters
        self.syllableWindowSize = 50 # In milliseconds
        self.syllablePeakMinimumDistance = 4
        self.syllablePeakMinimumWidth = 2
        self.syllablePitchDistanceTolerance = 4
        self.syllableZcrThreshold = 0.06

        # Filled pause parameters
        self.filledPauseWindowSize = 50 # In milliseconds
        self.filledPauseMinimumLength = 200 # In milliseconds
        self.filledPauseMinimumDistanceToPrevious = 1000 # In milliseconds
        self.filledPauseF1MaximumVariance = 60
        self.filledPauseF2MaximumVariance = 30
        self.filledPauseMaximumFormantDistance = 2000 # In Hz
        self.filledPauseMaximumSpectralFlatnessVariance = 0.001

        # Recording parameters
        self.recordingDeviceIndex = -1 # Default to asking user
        self.recordingBufferSize = 4096 # In samples
        self.recordingFormat = pyaudio.paInt16
        self.recordingChannels = 2



    def getEnergyFromAudio(self, audio):
        energy = featureModule.getEnergy(data= audio.data,
                                         sampleRate= audio.sampleRate,
                                         windowSize= self.energyWindowSize,
                                         stepSize= self.featureStepSize)
        return energy

    def getPitchFromAudio(self, audio, energy= None):
        if energy is None:
            energy = self.getEnergyFromAudio(audio)

        energyMinThreshold = featureModule.getEnergyMinimumThreshold(energy, self.energyThresholdRatio)
        fractionEnergyMinThreshold = energyMinThreshold / max(energy)

        pitches = featureModule.getPitch(data= audio.data,
                                         sampleRate= audio.sampleRate,
                                         stepSize= self.featureStepSize,
                                         silenceProportionThreshold= fractionEnergyMinThreshold,
                                         minimumRunLength= self.pitchMinimumRunLength)
        return pitches

    def getVoiceActivityFromAudio(self, audio, pitches= None):
        if pitches is None:
            pitches = self.getPitchFromAudio(audio)

        voiceActivity = featureModule.getVoiceActivity(data= audio.data,
                                                       sampleRate= audio.sampleRate,
                                                       pitchValues= pitches,
                                                       windowSize= self.voiceActivityWindowSize,
                                                       stepSize= self.featureStepSize,
                                                       useAdaptiveThresholds= self.voiceActivityIsAdaptive,
                                                       zcrThreshold= self.voiceActivityZCRThreshold,
                                                       energyPrimaryThreshold= self.voiceActivityEnergyThreshold,
                                                       pitchTolerance= self.voiceActivityPitchTolerance,
                                                       minimumRunLength= self.voiceActivityMinimumRunLength)
        return voiceActivity

    def getSyllablesFromAudio(self, audio, pitches= None):
        if pitches is None:
            pitches = self.getPitchFromAudio(audio)

        syllables, timeStamps = featureModule.getSyllables(data= audio.data,
                                               sampleRate= audio.sampleRate,
                                               pitchValues= pitches,
                                               windowSize= self.syllableWindowSize,
                                               stepSize= self.featureStepSize,
                                               energyPeakMinimumDistance= self.syllablePeakMinimumDistance,
                                               energyPeakMinimumWidth= self.syllablePeakMinimumWidth,
                                               pitchDistanceTolerance= self.syllablePitchDistanceTolerance,
                                               zcrThreshold= self.syllableZcrThreshold,
                                               energyThresholdRatio= self.energyThresholdRatio)
        return syllables, timeStamps

    def getFilledPausesFromAudio(self, audio):
        filledPauses, timeStamps = featureModule.getFilledPauses(data= audio.data,
                                                     sampleRate= audio.sampleRate,
                                                     windowSize= self.filledPauseWindowSize,
                                                     stepSize= self.featureStepSize,
                                                     minumumLength= self.filledPauseMinimumLength,
                                                     minimumDistanceToPrevious= self.filledPauseMinimumDistanceToPrevious,
                                                     F1MaximumVariance= self.filledPauseF1MaximumVariance,
                                                     F2MaximumVariance= self.filledPauseF2MaximumVariance,
                                                     maximumFormantDistance= self.filledPauseMaximumFormantDistance,
                                                     maximumSpectralFlatnessVariance= self.filledPauseMaximumSpectralFlatnessVariance,
                                                     energyThresholdRatio= self.energyThresholdRatio)
        return filledPauses, timeStamps



    # | Extracts features from audio and returns feature set.
    # | Parameters:
    # |   - audio: audio data to process
    # | Returns:
    # |   - Feature set
    def getFeaturesFromAudio(self, audio):
        features = featureModule.FeatureSet()

        totalStartTime = time.time()
        startTime = time.time()

        ### AMPLITUDE
        energy = self.getEnergyFromAudio(audio)

        if DEBUG:
            print("Time to get amplitude:", time.time() - startTime)
            startTime = time.time()

        ### PITCH
        pitches = self.getPitchFromAudio(audio, energy)

        if DEBUG:
            print("Time to get pitch:", time.time() - startTime)
            startTime = time.time()

        ### VAD
        voiceActivity = self.getVoiceActivityFromAudio(audio, pitches)

        if DEBUG:
            print("Time to get voice activity:", time.time() - startTime)
            startTime = time.time()

        ### SYLLABLES
        if "syllables" in self.features:
            syllables, _ = self.getSyllablesFromAudio(audio, pitches)

            if DEBUG:
                print("Time to get syllables:", time.time() - startTime)
                startTime = time.time()

        # ### FILLED PAUSES
        if "filledPauses" in self.features:
            filledPauses, _ = self.getFilledPausesFromAudio(audio)

            if DEBUG:
                print("Time to get filled pauses:", time.time() - startTime)
                print("Time to get features:", time.time() - totalStartTime)



        # Mask features with voice activity
        bufferFrames = int(self.voiceActivityMaskBufferSize / self.featureStepSize)
        mask = np.invert(featureModule.createBufferedBinaryArrayFromArray(voiceActivity.astype(bool), bufferFrames))

        if self.maskEnergyWithVoiceActivity:
            energy[mask] = 0

        if self.maskPitchWIthVoiceActivity:
            pitches[mask] = 0

        if self.maskSyllablesWithVoiceActivity:
            syllables[mask] = 0

        if self.maskFilledPausesWithVoiceActivity:
            filledPauses[mask] = 0



        # Calculate statistics and add to feature arrays
        ### AMPLITUDE
        averageEnergy = np.mean(energy)
        stDevEnergy = np.std(energy)

        features.meanIntensity = np.append(features.meanIntensity, averageEnergy)
        features.stDevIntensity = np.append(features.stDevIntensity, stDevEnergy)

        ### PITCH
        if max(pitches) > 0:
            pitches[pitches == 0] = np.nan

            averagePitch = np.nanmean(pitches)
            stDevPitch = np.nanstd(pitches)
        else:
            averagePitch = 0
            stDevPitch = 0

        features.meanPitch = np.append(features.meanPitch, averagePitch)
        features.stDevPitch = np.append(features.stDevPitch, stDevPitch)

        ### VOICE ACTIVITY
        averageVoiceActivity = np.mean(voiceActivity)
        stDevVoiceActivity = np.std(voiceActivity)

        features.meanVoiceActivity = np.append(features.meanVoiceActivity, averageVoiceActivity)
        features.stDevVoiceActivity = np.append(features.stDevVoiceActivity, stDevVoiceActivity)

        ### WORDS PER MINUTE
        if "syllables" in self.features:
            currentSyllablesPerSecond = int(sum(syllables))/self.lookBackSize

            features.syllablesPerSecond = np.append(features.syllablesPerSecond, currentSyllablesPerSecond)

        # FILLED PAUSES
        if "filledPauses" in self.features:
            features.filledPauses = np.append(features.filledPauses, int(sum(filledPauses)))

        return features

    # | Extracts features and returns array in accordance with Jamison's drawing
    # | Parameters:
    # |   - filePath: path to file to process
    # | Returns:
    # |   - Numpy array with features
    def getFeaturesFromFileUsingWindowing(self, filePath):
        # Read in the file
        audio = audioModule.Audio(filePath)
        if audio.numberOfChannels > 1:
            audio.makeMono()

        # Set up time tracker
        seconds = np.zeros(shape=0)

        # Set up arrays for features
        features = featureModule.FeatureSet()

        step = 0
        sampleStepSize = int(self.stepSize * audio.sampleRate)
        sampleLookBackSize = int(self.lookBackSize * audio.sampleRate)

        while step < audio.length:

            if self.printStatus:
                print("[", str(step/audio.length*100)[:4], "% ] Second", int(step/audio.sampleRate), "of", filePath, end="\r")

            # Keep track of what second we're in
            seconds = np.append(seconds,step/audio.sampleRate)

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

        # Pull all the feautures together in one array
        featureArray = np.vstack([seconds])
        for feature in self.features:
            if feature == "meanIntensity":
                featureArray = np.vstack([featureArray, features.meanIntensity])
            elif feature == "stDevIntensity":
                featureArray = np.vstack([featureArray, features.stDevIntensity])
            elif feature == "meanPitch":
                featureArray = np.vstack([featureArray, features.meanPitch])
            elif feature == "stDevPitch":
                featureArray = np.vstack([featureArray, features.stDevPitch])
            elif feature == "meanVoiceActivity":
                featureArray = np.vstack([featureArray, features.meanVoiceActivity])
            elif feature == "stDevVoiceActivity":
                featureArray = np.vstack([featureArray, features.stDevVoiceActivity])
            elif feature == "syllables":
                featureArray = np.vstack([featureArray, features.syllablesPerSecond])
            elif feature == "filledPauses":
                featureArray = np.vstack([featureArray, features.filledPauses])

        if self.printStatus :
            print("[ DONE ] Finished processing", filePath, "!")

        return featureArray

    # | Extracts features from all files in a directory and saves to numpy files.
    # | Parameters:
    # |   - inDirectory: directory for audio files
    # |   - outDirectory: directory for saving numpy files
    def createFeatureFilesFromDirectory(self, inDirectory, outDirectory):
        for featureName in self.features:
            assert featureName in FEATURE_NAMES, 'Invalid feature name in list.'

        with open(outDirectory + 'labels.csv', 'w') as outputFile:
            writer = csv.writer(outputFile)
            writer.writerow(["time"] + self.features)

        # Keep track of running stats
        startTime = time.time()
        count = 1

        audioFiles = inDirectory + "*.wav"

        for path in sorted(glob.iglob(audioFiles),reverse=False):
            # Communicate progress
            print("[ " + str(count) + "/" + str(len(sorted(glob.iglob(audioFiles)))) + " ] \tStarting:",path)

            featureArray = self.getFeaturesFromFileUsingWindowing(path)

            # Save the numpy array
            np.save(outDirectory + os.path.basename(path)[:-4], featureArray)

            # Crunch some numbers and communicate to the user
            timeElapsed = time.time() - startTime
            estimatedTimeRemaining = timeElapsed/count * (len(sorted(glob.iglob(audioFiles))) - count)
            print("\t\t" + str(timeElapsed/60) + " minutes elapsed. Estimated time remaining: " + str(estimatedTimeRemaining/60))

            count += 1

    # | Extracts features from live audio.
    def getFeaturesFromLiveInput(self):
        for featureName in self.features:
            assert featureName in FEATURE_NAMES, 'Invalid feature name in list.'

        # Controls the microphone and live input
        audioController = pyaudio.PyAudio()

        # Check if microphone paramter is set, default is -1
        microphoneIndex = self.recordingDeviceIndex

        # Check microphone paramter is valid
        if microphoneIndex < 0 or microphoneIndex >= audioController.get_device_count():
            # Use helper function to select the device
            microphoneIndex = audioModule.pickMicrophone(audioController)

        # Get the chosen device's sample rate
        sampleRate = int(audioController.get_device_info_by_index(microphoneIndex)["defaultSampleRate"])

        input("Press 'Enter' to start recording. Use keyboard interrupt to stop.")

        audioStream = audioController.open(format=self.recordingFormat,
                                           input_device_index=microphoneIndex,
                                           channels=self.recordingChannels,
                                           rate=sampleRate,
                                           input=True,
                                           frames_per_buffer=self.recordingBufferSize)

        print("\n\u001b[31m• Live\u001b[0m\r", end="\r")

        # Setup before recording starts
        data = np.zeros(shape=0)

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
                monoWaveData = np.mean(waveData,axis=1)

                # Add the just-read buffer to the current running array of audio data
                data = np.append(data, monoWaveData)

                # Once enough time has passed to include an entire window
                if data.size >= windowSampleSize:
                    # Chop out a window sized section of data
                    currentWindowSelection = data[0:windowSampleSize]

                    # Create an object to pass around
                    audio = audioModule.Audio(data=currentWindowSelection)
                    audio.sampleRate = sampleRate
                    audio.numberOfChannels = 2 # TODO: Check if this should be 1

                    # Get all features for this window of audio
                    features = self.getFeaturesFromAudio(audio)

                    print("\u001b[31m• Live\u001b[0m", str((time.time() - startTime))[:5], "elapsed.",
                          "Mean pitch:", features.meanPitch[0],
                          "Voice activity:", features.meanVoiceActivity[0],
                          "Speech Rate:", features.syllablesPerSecond[0],
                          "Filled pauses:", features.filledPauses[0],
                          "Time:", time.time() - startTime,
                          end="                    \r")

                    # Reduce the size of the audio data array to move the beggining
                    # forward by one step size so the next window is offset by this amount
                    data = data[stepSampleSize:]

        # User stops with ctrl + c
        except KeyboardInterrupt:
            print("\rStopped!                                                                     ")

        # Clean up audio session
        audioStream.stop_stream()
        audioStream.close()
        audioController.terminate()
