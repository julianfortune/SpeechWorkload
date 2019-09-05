#
# Created on Apr 23, 2019
#
# @author: Julian Fortune
# @Description: Interface for extracting feature arrays.
#

import os, glob, sys # file io
import time
import numpy as np
import pandas as pd
from datetime import datetime

import pyaudio # microphone io
import wavio # microphone decoding
import threading # Background processing
import multiprocessing

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

        # Audio processing parameters
        self.removeDCOffsetFromAudio = True

        # Windowing parameters
        self.stepSize = 1 # Time interval between extracting features, in seconds.
        self.lookBackSize = 30 # Time interval to examine looking backward for features, in seconds.

        # Parameters for all features
        self.features = ["meanIntensity", "stDevIntensity", "meanPitch", "stDevPitch", "stDevVoiceActivity", "meanVoiceActivity", "syllablesPerSecond", "filledPauses"]
        self.featureStepSize = 10 # Step size for all features, in milliseconds.
        self.energyThresholdRatio = 4

        self.voiceActivityMaskBufferSize = 100 # In milliseconds

        self.maskEnergyWithVoiceActivity = False
        self.maskPitchWIthVoiceActivity = True
        self.maskSyllablesWithVoiceActivity = True
        self.maskFilledPausesWithVoiceActivity = False

        # Energy parameters
        self.energyWindowSize = 50

        # Pitch parameters
        self.pitchMinimumRunLength = 4

        # Voice activity Parameters
        self.voiceActivityIsAdaptive = True
        self.voiceActivityWindowSize = 50 # In milliseconds
        self.voiceActivityZCRMaximumThreshold = 0.04 # Originally 0.06
        self.voiceActivityZCRMinimumThreshold = 0.008
        self.voiceActivityEnergyThreshold = 40
        self.voiceActivityPitchTolerance = 8
        self.voiceActivityMinimumRunLength = 10

        # Syllable detection parameters
        self.syllableWindowSize = 50 # In milliseconds
        self.syllablePeakMinimumDistance = 4
        self.syllablePeakMinimumWidth = 2 # Maybe 4 ?
        self.syllablePeakMinimumProminence = None # Maybe 75 ?
        self.syllablePitchDistanceTolerance = 4
        self.syllableZcrThreshold = 0.06

        # Filled pause parameters
        self.filledPauseWindowSize = 50 # In milliseconds
        self.filledPauseMinimumLength = 250 # In milliseconds
        self.filledPauseMinimumDistanceToPrevious = 1000 # In milliseconds
        self.filledPauseF1MaximumVariance = 60
        self.filledPauseF2MaximumVariance = 30
        self.filledPauseMaximumFormantDistance = 1000 # In Hz

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
                                                       zcrMaximumThreshold= self.voiceActivityZCRMaximumThreshold,
                                                       zcrMinimumThreshold= self.voiceActivityZCRMinimumThreshold,
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
                                               energyPeakMinimumProminence = self.syllablePeakMinimumProminence,
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
                                                     energyThresholdRatio= self.energyThresholdRatio)
        return filledPauses, timeStamps



    # | Extracts features from audio and returns feature set.
    # | Parameters:
    # |   - audio: audio data to process
    # | Returns:
    # |   - Feature set
    def getFeaturesFromAudio(self, audio, shouldTime= False):
        features = featureModule.FeatureSet()

        times = {}

        if shouldTime:
            totalStartTime = time.time()
            startTime = time.time()

        # Get amplitude envelope feature.
        energy = self.getEnergyFromAudio(audio)

        if __debug__:
            if shouldTime:
                times["intensity"] = time.time() - startTime
                startTime = time.time()

        # Get pitch feature.
        pitches = self.getPitchFromAudio(audio, energy)

        if __debug__:
            if shouldTime:
                times["pitch"] = time.time() - startTime
                startTime = time.time()

        # Get voice activity feature.
        voiceActivity = self.getVoiceActivityFromAudio(audio, pitches)

        if __debug__:
            if shouldTime:
                times["voiceActivity"] = time.time() - startTime
                startTime = time.time()

        # Get syllables feature if needed as binary array for easy masking.
        if "syllablesPerSecond" in self.features:
            syllables, _ = self.getSyllablesFromAudio(audio, pitches)

            if __debug__:
                if shouldTime:
                    times["syllables"] = time.time() - startTime
                    startTime = time.time()

        # Get filled pauses feature if needed as binary array for easy masking.
        if "filledPauses" in self.features:
            filledPauses, _ = self.getFilledPausesFromAudio(audio)

            if __debug__:
                if shouldTime:
                    times["filledPauses"] = time.time() - startTime
                    startTime = time.time()



        # Expand voice activity to have a margin for error to catch speech features
        # and create boolean array with True for no activity to set no activity
        # to all zeros.
        bufferFrames = int(self.voiceActivityMaskBufferSize / self.featureStepSize)
        mask = np.invert(featureModule.createBufferedBinaryArrayFromArray(voiceActivity.astype(bool), bufferFrames))

        # Mask features with voice activity but setting regions with no voice
        # activity (incl. buffered margin of error) to zero.
        if self.maskEnergyWithVoiceActivity:
            # Prevent mismatches in the lengths of the arrays
            energy = energy[:len(mask)]
            energy[mask[:len(energy)]] = 0

        if self.maskPitchWIthVoiceActivity:
            pitches = pitches[:len(mask)]
            pitches[mask[:len(pitches)]] = np.nan

        if self.maskSyllablesWithVoiceActivity:
            syllables = syllables[:len(mask)]
            syllables[mask[:len(syllables)]] = 0

        if self.maskFilledPausesWithVoiceActivity:
            filledPauses = filledPauses[:len(mask)]
            filledPauses[mask[:len(filledPauses)]] = 0



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
        if "syllablesPerSecond" in self.features:
            currentSyllablesPerSecond = int(sum(syllables))/self.lookBackSize

            features.syllablesPerSecond = np.append(features.syllablesPerSecond, currentSyllablesPerSecond)

        # FILLED PAUSES
        if "filledPauses" in self.features:
            features.filledPauses = np.append(features.filledPauses, int(sum(filledPauses)))

        if __debug__:
            if shouldTime:
                times["processingFeatures"] = time.time() - startTime

        if shouldTime:
            return features, times
        else:
            return features

    # | Extracts features and returns array in accordance with Jamison's drawing
    # | Parameters:
    # |   - filePath: path to file to process
    # | Returns:
    # |   - Numpy array with features
    def getFeaturesFromFileUsingWindowing(self, filePath, shouldTime=True):
        name = os.path.basename(filePath)
        timingData = pd.DataFrame([], columns=['intensity', 'pitch', 'voiceActivity', 'syllables', 'filledPauses', 'processingFeatures'])
        timingData.index.name = 'time'

        # Read in the file
        audio = audioModule.Audio(filePath)
        if audio.numberOfChannels > 1:
            audio.makeMono()

        if self.removeDCOffsetFromAudio:
            audio.unBias() # Move the center of the audio to 0

        startTime = time.time()

        # Set up time tracker
        seconds = np.zeros(shape=0)

        # Set up arrays for features
        features = featureModule.FeatureSet()

        step = 0
        sampleStepSize = int(self.stepSize * audio.sampleRate)
        sampleLookBackSize = int(self.lookBackSize * audio.sampleRate)

        while step < audio.length:

            if self.printStatus:
                print("[", str(step/audio.length*100)[:4], "% ]",
                      "Second", int(step/audio.sampleRate), "of", name,
                      end="")
                if int(step/audio.sampleRate) > self.lookBackSize:
                    print(" - Time per second:", (time.time() - startTime)/(int(step/audio.sampleRate) - self.lookBackSize), end="\r")
                else:
                    print(end="\r")

            # Keep track of what second we're in
            seconds = np.append(seconds,step/audio.sampleRate)

            # Look backward to calculate features over long term
            if step + sampleStepSize - sampleLookBackSize > 0:

                currentWindow = audioModule.Audio(data=audio.data[step + sampleStepSize - sampleLookBackSize:step + sampleStepSize])
                currentWindow.sampleRate = audio.sampleRate

                if shouldTime:
                    currentFeatures, timesDictionary = self.getFeaturesFromAudio(currentWindow, shouldTime= True)
                else:
                    currentFeatures = self.getFeaturesFromAudio(currentWindow)

                features.append(currentFeatures)

                #Handle timing
                timingData = timingData.append(pd.DataFrame(timesDictionary, index=[int(step/audio.sampleRate)]))

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
            elif feature == "syllablesPerSecond":
                featureArray = np.vstack([featureArray, features.syllablesPerSecond])
            elif feature == "filledPauses":
                featureArray = np.vstack([featureArray, features.filledPauses])

        if self.printStatus :
            print("[ DONE ] Finished processing", filePath, "!")

        if shouldTime:
            return featureArray, timingData
        else:
            return featureArray

    # | Write parameters used to generate the features.
    def saveInfoToFile(self, directory):
        with open(directory + 'about.txt', 'w+') as aboutFile:
            aboutFile.write("Started " + str(datetime.today().strftime('%Y-%m-%d')) + "\n")
            aboutFile.write("" + "\n")
            aboutFile.write("Description --------------------------------------------------" + "\n")
            aboutFile.write("" + "\n")
            aboutFile.write("Parameters ---------------------------------------------------" + "\n")
            aboutFile.write("" + "\n")

            # Audio processing parameters
            aboutFile.write("removeDCOffsetFromAudio (bool) = " + str(self.removeDCOffsetFromAudio) + "\n")
            aboutFile.write("" + "\n")

            # Windowing parameters
            aboutFile.write("stepSize (seconds) = " + str(self.stepSize) + "\n")
            aboutFile.write("lookBackSize (seconds) = " + str(self.lookBackSize) + "\n")
            aboutFile.write("" + "\n")

            # Parameters for all features
            aboutFile.write("features = " + str(self.features) + "\n")
            aboutFile.write("featureStepSize (milliseconds) = " + str(self.featureStepSize) + "\n")
            aboutFile.write("energyThresholdRatio = " + str(self.energyThresholdRatio) + "\n")
            aboutFile.write("" + "\n")

            aboutFile.write("voiceActivityMaskBufferSize (milliseconds) = " + str(self.voiceActivityMaskBufferSize) + "\n")
            aboutFile.write("" + "\n")

            aboutFile.write("maskEnergyWithVoiceActivity = " + str(self.maskEnergyWithVoiceActivity) + "\n")
            aboutFile.write("maskPitchWIthVoiceActivity = " + str(self.maskPitchWIthVoiceActivity) + "\n")
            aboutFile.write("maskSyllablesWithVoiceActivity = " + str(self.maskSyllablesWithVoiceActivity) + "\n")
            aboutFile.write("maskFilledPausesWithVoiceActivity = " + str(self.maskFilledPausesWithVoiceActivity) + "\n")
            aboutFile.write("" + "\n")

            # Pitch parameters
            aboutFile.write("energyWindowSize = " + str(self.energyWindowSize) + "\n")
            aboutFile.write("" + "\n")

            # Pitch parameters
            aboutFile.write("pitchMinimumRunLength = " + str(self.pitchMinimumRunLength) + "\n")
            aboutFile.write("" + "\n")

            # Voice activity Parameters
            aboutFile.write("voiceActivityIsAdaptive = " + str(self.voiceActivityIsAdaptive) + "\n")
            aboutFile.write("voiceActivityMaskBufferSize (milliseconds) = " + str(self.voiceActivityWindowSize) + "\n")
            aboutFile.write("voiceActivityZCRMaximumThreshold (milliseconds) = " + str(self.voiceActivityZCRMaximumThreshold) + "\n")
            aboutFile.write("voiceActivityZCRMinimumThreshold (milliseconds) = " + str(self.voiceActivityZCRMinimumThreshold) + "\n")
            aboutFile.write("voiceActivityEnergyThreshold (milliseconds) = " + str(self.voiceActivityEnergyThreshold) + "\n")
            aboutFile.write("voiceActivityPitchTolerance (milliseconds) = " + str(self.voiceActivityPitchTolerance) + "\n")
            aboutFile.write("voiceActivityMinimumRunLength (milliseconds) = " + str(self.voiceActivityMinimumRunLength) + "\n")
            aboutFile.write("" + "\n")

            # Syllable detection parameters
            aboutFile.write("syllableWindowSize (milliseconds) = " + str(self.syllableWindowSize) + "\n")
            aboutFile.write("syllablePeakMinimumDistance = " + str(self.syllablePeakMinimumDistance) + "\n")
            aboutFile.write("syllablePeakMinimumWidth = " + str(self.syllablePeakMinimumWidth) + "\n")
            aboutFile.write("syllablePitchDistanceTolerance = " + str(self.syllablePitchDistanceTolerance) + "\n")
            aboutFile.write("syllableZcrThreshold = " + str(self.syllableZcrThreshold) + "\n")
            aboutFile.write("" + "\n")

            # Filled pause parameters
            aboutFile.write("filledPauseWindowSize (milliseconds) = " + str(self.filledPauseWindowSize) + "\n")
            aboutFile.write("filledPauseMinimumLength (milliseconds) = " + str(self.filledPauseMinimumLength) + "\n")
            aboutFile.write("filledPauseMinimumDistanceToPrevious (milliseconds) = " + str(self.filledPauseMinimumDistanceToPrevious) + "\n")
            aboutFile.write("filledPauseF1MaximumVariance = " + str(self.filledPauseF1MaximumVariance) + "\n")
            aboutFile.write("filledPauseF2MaximumVariance = " + str(self.filledPauseF2MaximumVariance) + "\n")
            aboutFile.write("filledPauseMaximumFormantDistance (Hz) = " + str(self.filledPauseMaximumFormantDistance) + "\n")
            aboutFile.write("" + "\n")

            # Recording parameters
            aboutFile.write("recordingDeviceIndex = " + str(self.recordingDeviceIndex) + "\n")
            aboutFile.write("recordingBufferSize (samples) = " + str(self.recordingBufferSize) + "\n")
            aboutFile.write("recordingFormat = " + str(self.recordingFormat) + "\n")
            aboutFile.write("recordingChannels = " + str(self.recordingChannels) + "\n")
            aboutFile.write("" + "\n")


    # | Extracts features from all files in a directory and saves to numpy files.
    # | Parameters:
    # |   - inDirectory: directory for audio files
    # |   - outDirectory: directory for saving numpy files
    def createFeatureFilesFromDirectory(self, inDirectory, outDirectory, saveRunTimes=False):
        for featureName in self.features:
            assert featureName in FEATURE_NAMES, 'Invalid feature name in list.'

        self.saveInfoToFile(outDirectory)

        # Keep track of running stats
        startTime = time.time()
        count = 1
        processedCount = 1

        audioFiles = inDirectory + "*.wav"
        featuresFiles = list(glob.iglob(outDirectory + "*.csv"))

        for path in sorted(glob.iglob(audioFiles),reverse=False):
            name = os.path.basename(path)[:-4]

            # Check if features already made
            if not (outDirectory + name + ".csv") in featuresFiles:
                # Communicate progress
                print("[ " + str(count) + "/" + str(len(sorted(glob.iglob(audioFiles)))) + " ] \tStarting:",path)

                if saveRunTimes:
                    featureArray, runTimeData = self.getFeaturesFromFileUsingWindowing(path, shouldTime= saveRunTimes)
                    print(saveRunTimes)
                    runTimeData.to_csv(outDirectory + os.path.basename(path)[:-4] + "-run_time.csv")
                else:
                    featureArray = self.getFeaturesFromFileUsingWindowing(path)

                # Save the numpy array by swapping into row major and saving as a
                # pandas-ready csv.
                featureArray = np.swapaxes(featureArray, 0, 1)
                frame = pd.DataFrame(featureArray, columns= ["time"] + self.features)
                frame.to_csv(outDirectory + os.path.basename(path)[:-4] + ".csv", index= False)

                # Crunch some numbers and communicate to the user
                timeElapsed = time.time() - startTime
                estimatedTimeRemaining = timeElapsed/processedCount * (len(sorted(glob.iglob(audioFiles))) - processedCount)
                print("\t\t" + str(timeElapsed/60) + " minutes elapsed. Estimated time remaining: " + str(estimatedTimeRemaining/60))

                processedCount += 1
            else:
                # Communicate skipping
                print("[ " + str(count) + "/" + str(len(sorted(glob.iglob(audioFiles)))) + " ] \tSkipping:",path)

            count += 1



    def getFeaturesInBackground(self, segment, sampleRate, startTime, featureExtractionStartTime, printLock):
        # Create an object to pass around
        audio = audioModule.Audio(data=segment)
        audio.sampleRate = sampleRate
        audio.numberOfChannels = 2 # TODO: Check if this should be 1

        # Get all features for this window of audio
        features = self.getFeaturesFromAudio(audio)

        with printLock:
            print("\u001b[31m• Live\u001b[0m", str((time.time() - startTime))[:5], "elapsed.",
                  "Mean pitch:", round(features.meanPitch[0],2),
                  "Voice activity:", round(features.meanVoiceActivity[0],2),
                  "Speech Rate:", round(features.syllablesPerSecond[0],2),
                  "Filled pauses:", round(features.filledPauses[0],2),
                  "Time to run:", round(time.time() - featureExtractionStartTime,4),
                  end="                    \r")

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
        runTime = time.time()
        printLock = threading.Lock()

        try:
            # Record until the user stops
            while True:
                # Read in from microphone
                buffer = audioStream.read(self.recordingBufferSize)
                # print("reading from buffer", len(buffer), data.shape, time.time() - runTime)
                # runTime = time.time()

                # Convert to mono float data
                waveData = wavio._wav2array(self.recordingChannels, audioController.get_sample_size(self.recordingFormat), buffer)
                monoWaveData = np.mean(waveData, axis=1)

                # Add the just-read buffer to the current running array of audio data
                data = np.append(data, monoWaveData)

                # Once enough time has passed to include an entire window
                if data.size >= windowSampleSize and data.size >= stepSampleSize:
                    # Chop out a window sized section of data
                    currentWindowSelection = data[0:windowSampleSize]

                    featureThread = multiprocessing.Process(target=self.getFeaturesInBackground, args=(currentWindowSelection, sampleRate, startTime, time.time(), printLock,))
                    featureThread.start()

                    # Reduce the size of the audio data array to move the beggining
                    # forward by one step size so the next window is offset by this amount
                    data = data[stepSampleSize:]

        # User stops with ctrl + c
        except KeyboardInterrupt:
            print("    \nStopped!                                                                     ")

        # Clean up audio session
        audioStream.stop_stream()
        audioStream.close()
        audioController.terminate()
