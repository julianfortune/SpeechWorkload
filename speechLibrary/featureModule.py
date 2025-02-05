#
# Created on Apr 23, 2019
#
# @author: Julian Fortune
# @Description: Functions for extracting features from audio data.
#

import math
import time

import librosa
import matplotlib.pyplot as plt
import numpy as np
import parselmouth  # Formants
import scipy.signal  # Extracts power spectrum

DEBUG_VOICE_SYLLABLE_GRAPH = False
DEBUG_VOICE_ACTIVITY_GRAPH = False
DEBUG_FILLED_PAUSE_GRAPH = False
DEBUG_TIME = False

class FeatureSet:

    # TODO: Redo to use classic lists and convert at very end to save on
    # re-allocation/copying time

    def __init__(self):
        self.syllablesPerSecond = np.zeros(shape=0)
        self.meanVoiceActivity = np.zeros(shape=0)
        self.stDevVoiceActivity = np.zeros(shape=0)
        self.meanPitch = np.zeros(shape=0)
        self.stDevPitch = np.zeros(shape=0)
        self.meanIntensity = np.zeros(shape=0)
        self.stDevIntensity = np.zeros(shape=0)
        self.filledPauses = np.zeros(shape=0)

    def appendAllZeros(self):
        self.syllablesPerSecond = np.append(self.syllablesPerSecond, 0)
        self.meanVoiceActivity = np.append(self.meanVoiceActivity, 0)
        self.stDevVoiceActivity = np.append(self.stDevVoiceActivity, 0)
        self.meanPitch = np.append(self.meanPitch, 0)
        self.stDevPitch = np.append(self.stDevPitch, 0)
        self.meanIntensity = np.append(self.meanIntensity, 0)
        self.stDevIntensity = np.append(self.stDevIntensity, 0)
        self.filledPauses = np.append(self.filledPauses, 0)

    def append(self, secondFeatureSet):
        self.syllablesPerSecond = np.append(self.syllablesPerSecond, secondFeatureSet.syllablesPerSecond)
        self.meanVoiceActivity = np.append(self.meanVoiceActivity, secondFeatureSet.meanVoiceActivity)
        self.stDevVoiceActivity = np.append(self.stDevVoiceActivity, secondFeatureSet.stDevVoiceActivity)
        self.meanPitch = np.append(self.meanPitch, secondFeatureSet.meanPitch)
        self.stDevPitch = np.append(self.stDevPitch, secondFeatureSet.stDevPitch)
        self.meanIntensity = np.append(self.meanIntensity, secondFeatureSet.meanIntensity)
        self.stDevIntensity = np.append(self.stDevIntensity, secondFeatureSet.stDevIntensity)
        self.filledPauses = np.append(self.filledPauses, secondFeatureSet.filledPauses)



# | Removes consecutive values in runs less than the minimum length.
def removeSmallRunsOfValues(npArray, minimumLength):
    currentlyInARun = False
    runStartingIndex = 0

    for index in range(0, len(npArray)):
        if npArray[index] != 0 and not currentlyInARun:
            currentlyInARun = True
            runStartingIndex = index
        if npArray[index] == 0 and currentlyInARun:
            currentlyInARun = False
            lengthOfRun = index - runStartingIndex
            if lengthOfRun < minimumLength:
                np.put(npArray, range(runStartingIndex, index + 1), 0)

# | Adds extra binary True values surrounding existing True values by the number
# | of frames specified.
def createBufferedBinaryArrayFromArray(npArray, frames):
    bufferedArray = np.full(len(npArray), False)

    for index in range(0, len(npArray)):
        start = index - frames
        end = index + frames + 1

        if start < 0:
            start = 0
        if end > len(npArray) - 1:
            end = len(npArray) - 1

        if True in npArray[start:end]:
            bufferedArray[index] = True

    return bufferedArray

# | Checks surrounding values in an array around the index to check if
# | any of them are above the threshold.
def aboveThresholdWithinTolerance(data, indexInQuestion, threshold, tolerance):
    for index in range(indexInQuestion - tolerance,indexInQuestion + tolerance):
        if index >= 0 and index < len(data):
            if data[index] > threshold:
                return True
    return False

# | Returns the minimum energy threshold using custom metric.
def getEnergyMinimumThreshold(energy, signalToNoiseRatio):
    return np.percentile(energy, 10) * signalToNoiseRatio

# | Returns energy values using librosa RMS.
def getEnergy(data, sampleRate, windowSize, stepSize):
    windowSizeInSamples = int(sampleRate / 1000 * windowSize)
    stepSizeInSamples = int(sampleRate / 1000 * stepSize)

    energy = librosa.feature.rms(data, frame_length=windowSizeInSamples, hop_length=stepSizeInSamples)[0]
    return energy

# | Returns energy values using short term energy calculation.
def getShortTermEnergy(data, sampleRate, windowSize, stepSize):
    windowSizeInSamples = int(sampleRate / 1000 * windowSize)
    stepSizeInSamples = int(sampleRate / 1000 * stepSize)

    # Pad the end of the array
    data = np.pad(data, (0, int(windowSizeInSamples - stepSizeInSamples)), mode='constant')

    # Create frames using optimized algorithm
    framedData = librosa.util.frame(data,
                                    frame_length=windowSizeInSamples,
                                    hop_length=stepSizeInSamples)

    shortTermEnergy = np.sum((framedData)**2, axis=0, keepdims=True)[0]

    return shortTermEnergy

# | Returns energy values using RMS calculation.
def getRMSIntensity(data, sampleRate, windowSize, stepSize):
    windowSizeInSamples = int(sampleRate / 1000 * windowSize)
    stepSizeInSamples = int(sampleRate / 1000 * stepSize)

    # Pad the end of the array
    data = np.pad(data, (0, int(windowSizeInSamples)), mode='constant')

    # Create frames using optimized algorithm
    framedData = librosa.util.frame(data,
                                    frame_length=windowSizeInSamples,
                                    hop_length=stepSizeInSamples)

    rms = np.sqrt(np.mean((framedData)**2, axis=0, keepdims=True))[0]

    return rms

# | Returns pitch values from PRAAT to_pitch_ac.
def getPitch(data, sampleRate, stepSize, silenceProportionThreshold, minimumRunLength):
    # Convert to the parselmouth custom sound type (req'd for formant function).
    parselSound = parselmouth.Sound(values=data, sampling_frequency=sampleRate)

    # Get the pitch values using PRAAT auto-correlation.
    pitchData = parselSound.to_pitch_ac(time_step=stepSize/1000,
                                        pitch_ceiling=400.0,
                                        silence_threshold=silenceProportionThreshold)
    pitchValues = pitchData.selected_array['frequency']
    removeSmallRunsOfValues(pitchValues, minimumRunLength)

    return np.array(pitchValues)

# | Returns the first two formants from a piece of audio.
def getFormants(data, sampleRate, windowSize, stepSize):
    # Convert to the parselmouth custom sound type (req'd for formant function).
    parselSound = parselmouth.Sound(values=data, sampling_frequency=sampleRate)

    # Produce a formant object from this data.
    formantData = parselSound.to_formant_burg(window_length=windowSize, time_step=stepSize)

    # API for parselmouth doesn't explain how to extract the formatData without querying for everything manually.
    firstFormant = []
    secondFormant = []

    times = np.arange(0, len(data)/sampleRate, stepSize) # In seconds

    for timeStamp in times:
        firstFormantValue = formantData.get_value_at_time(1, timeStamp)
        firstFormant.append(firstFormantValue)

        secondFormantValue = formantData.get_value_at_time(2, timeStamp)
        secondFormant.append(secondFormantValue)

    return np.array(firstFormant), np.array(secondFormant)

# | Returns the indices of syllables in audio data.
def getSyllables(data, sampleRate, pitchValues, windowSize, stepSize, energyPeakMinimumDistance, energyPeakMinimumWidth, energyPeakMinimumProminence, pitchDistanceTolerance, zcrThreshold, energyThresholdRatio):
    # Convert window and step sizes to samples for Librosa.
    windowSizeInSamples = int(sampleRate / 1000 * windowSize)
    stepSizeInSamples = int(sampleRate / 1000 * stepSize)

    # Empty integer array
    numberOfSteps = int(len(data) / stepSizeInSamples)
    syllables = np.full(numberOfSteps, 0)

    # Get energy.
    energy = getEnergy(data, sampleRate, windowSize, stepSize)

    # Get energy threshold
    energyMinThreshold = getEnergyMinimumThreshold(energy, energyThresholdRatio)

    # Get zero-crossing Rate
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=windowSizeInSamples, hop_length=stepSizeInSamples)[0]

    # Identify peaks in energy.
    peaks, _ = scipy.signal.find_peaks(energy,
                                       height=energyMinThreshold,
                                       distance=energyPeakMinimumDistance,
                                       width=energyPeakMinimumWidth,
                                       prominence=energyPeakMinimumProminence)

    validPeaks = []

    # Remove candidate peaks that don't meet voicing requirements.
    for i in range(0,len(peaks)):
        if zcr[peaks[i]] < zcrThreshold and aboveThresholdWithinTolerance(data=pitchValues,
                                                                          indexInQuestion=peaks[i],
                                                                          threshold=0,
                                                                          tolerance=pitchDistanceTolerance):
            validPeaks = np.append(validPeaks, peaks[i])
            syllables[peaks[i]] = 1

    if __debug__:
        if DEBUG_VOICE_SYLLABLE_GRAPH:
            # Show graph if debugging
            times = np.arange(0, len(data)/sampleRate, stepSize/1000)
            energyTimes = np.arange(0, len(data)/sampleRate, stepSize/1000)[:len(energy)]
            zcrTimes = np.arange(0, len(data)/sampleRate, stepSize/1000)[:len(zcr)]
            pitchTimes = np.arange(0, len(data)/sampleRate, stepSize/1000)[:len(pitchValues)]
            pitchValues[pitchValues == 0] = np.nan

            syllableMarkers = np.full(len(validPeaks), 0)

            plt.figure(figsize=[16, 8])
            plt.plot(energyTimes, energy / 100000000, zcrTimes, zcr * 10000, pitchTimes, pitchValues)
            plt.plot(np.array(validPeaks) * stepSizeInSamples / sampleRate, syllableMarkers, 'go')
            plt.show()
            plt.close()

    # Return syllables & candidate peaks that didn't meet voicing requirements.
    return syllables, np.array(validPeaks) * stepSizeInSamples / sampleRate

# | Returns the voice activity (each v_i in V ∈ {0,1}) using an adaptive algorithm from "A Simple but Efficient...".
def getVoiceActivity(data, sampleRate, pitchValues, windowSize, stepSize, useAdaptiveThresholds, zcrMaximumThreshold, zcrMinimumThreshold, energyPrimaryThreshold, pitchTolerance, minimumRunLength):
    # Convert window and step sizes to samples for Librosa.
    windowSizeInSamples = int(sampleRate / 1000 * windowSize)
    stepSizeInSamples = int(sampleRate / 1000 * stepSize)

    # Get energy and zero-crossing rate.
    energy = getShortTermEnergy(data, sampleRate, windowSize, stepSize)
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=windowSizeInSamples, hop_length=stepSizeInSamples)[0]

    # Move the thresholds if needed.
    if useAdaptiveThresholds:
        minEnergy = np.mean(energy[0:30])
        energyThreshold = energyPrimaryThreshold * math.log(minEnergy)
    else:
        energyThreshold = energyPrimaryThreshold

    voiceActivity = []
    silenceCount = 0

    for i in range(0, len(energy)):
        currentActivity = 0

        if (zcr[i] < zcrMaximumThreshold and zcr[i] > zcrMinimumThreshold and
            energy[i] > energyThreshold and aboveThresholdWithinTolerance(data= pitchValues,
                                                                          indexInQuestion= i,
                                                                          threshold= 0,
                                                                          tolerance= pitchTolerance)):

            currentActivity = 1 # Voice acitivty present
        else:
            silenceCount += 1

        voiceActivity = np.append(voiceActivity, currentActivity) # No voice acitivty present

        if useAdaptiveThresholds:
            minEnergy = ( (silenceCount * minEnergy) + energy[i] ) / ( silenceCount + 1 )
            energyThreshold = energyPrimaryThreshold * math.log(minEnergy)

    removeSmallRunsOfValues(voiceActivity, minimumRunLength)

    if __debug__:
        if DEBUG_VOICE_ACTIVITY_GRAPH:
            # Show graph if debugging
            energyTimes = np.arange(0, len(data)/sampleRate, stepSize/1000)[:len(energy)]
            zcrTimes = np.arange(0, len(data)/sampleRate, stepSize/1000)[:len(zcr)]
            pitchTimes = np.arange(0, len(data)/sampleRate, stepSize/1000)[:len(pitchValues)]
            plotVoiceActivity = np.copy(voiceActivity)
            plotVoiceActivity[plotVoiceActivity == 0] = np.nan
            pitchValues[pitchValues == 0] = np.nan

            plt.figure(figsize=[16, 8])
            plt.plot(energyTimes, energy / 100000000, zcrTimes, zcr * 10000, pitchTimes, pitchValues)
            plt.plot(energyTimes, plotVoiceActivity * -100)
            plt.show()
            plt.close()

    return voiceActivity

# | Returns an array of timestamps where filled pauses were detected.
def getFilledPauses(data, sampleRate, windowSize, stepSize, minumumLength, minimumDistanceToPrevious, F1MaximumVariance, F2MaximumVariance, maximumFormantDistance, energyThresholdRatio):
    # Convert window and step sizes to samples for Librosa and to prevent rounding issues with RMSE.
    windowSizeInSamples = int(sampleRate / 1000 * windowSize)
    stepSizeInSamples = int(sampleRate / 1000 * stepSize)

    timeStamps = []

    # The number of steps in the feature arrays.
    numberOfSteps = int(len(data) / stepSizeInSamples)
    # Empty integer array
    filledPauses = np.full(numberOfSteps, 0)

    # Get energy, first and second formants (F1 & F2), and spectral flatness.
    energy = getEnergy(data, sampleRate, windowSize, stepSize)
    firstFormant, secondFormant = getFormants(data, sampleRate,
                                              windowSizeInSamples/sampleRate,
                                              stepSizeInSamples/sampleRate)

    energyThreshold = getEnergyMinimumThreshold(energy, energyThresholdRatio)

    # Used for finding time stamp.
    times = np.arange(0, len(data)/sampleRate, stepSizeInSamples/sampleRate) # seconds

    # The number of steps in the feature arrays that make up a single window for checking for utterances.
    utteranceWindowSize = int(minumumLength / 1000 * sampleRate / stepSizeInSamples)

    fillerUtteranceInitiated = False

    # Step through each data point in formant and energy arrays and check for
    # filler utterances over the next 'minimumLength' size window of features.
    for step in range(0, numberOfSteps-utteranceWindowSize):
        start = step
        end = step+utteranceWindowSize

        firstFormantVariance = np.std(firstFormant[start:end])
        secondFormantVariance = np.std(secondFormant[start:end])
        averageFormantDistance = np.mean(secondFormant[start:end] - firstFormant[start:end])

        averageEnergy = np.mean(energy[start:end])

        # Check for any filled pauses immediately before the current window
        previousFilledPause = 0
        if len(timeStamps) > 0:
            previousFilledPause = timeStamps[-1]
        else:
            previousFilledPause = -10
        distanceToPreviousFilledPause = times[step] - previousFilledPause

        # Identify filled pauses
        if (firstFormantVariance <= F1MaximumVariance and
            secondFormantVariance <= F2MaximumVariance and
            averageEnergy > energyThreshold and
            distanceToPreviousFilledPause > minimumDistanceToPrevious/1000 and
            averageFormantDistance < maximumFormantDistance):
            # Prevent an utterance from being detected many times.
            if fillerUtteranceInitiated == False:
                fillerUtteranceInitiated = True

                timeStamps.append(times[step])
                filledPauses[step] = 1
        else:
            fillerUtteranceInitiated = False

    if __debug__:
        if DEBUG_FILLED_PAUSE_GRAPH:
            # Show graph if debugging
            filledPausesMarkers = np.full(len(timeStamps), 0)
            energyTimes = np.arange(0, len(data)/sampleRate, stepSize/1000)[:len(energy)]

            plt.figure(figsize=[16, 8])
            plt.plot(energyTimes, energy[:len(energyTimes)] / 10)
            plt.plot(times, firstFormant, times, secondFormant, np.array(timeStamps), filledPausesMarkers, 'go')
            plt.show()
            plt.close()

    return filledPauses, np.array(timeStamps)

# # | [WIP] The start of an implementation of Garg and Ward's filled pause algorithm.
# def getFilledPausesGargAndWard(data, sampleRate):
#     # Convert window and step sizes to samples for Librosa and to prevent rounding issues with RMSE.
#     windowSizeInSamples = int(sampleRate / 1000 * windowSize)
#     stepSizeInSamples = int(sampleRate / 1000 * stepSize)

#     timeStamps = []

#     # The number of steps in the feature arrays.
#     numberOfSteps = int(len(data) / stepSizeInSamples)

#     # Convert to the parselmouth custom sound type (req'd for formant function).
#     parselSound = parselmouth.Sound(values=data, sampling_frequency=sampleRate)

#     # Get the pitch values using PRAAT auto-correlation.
#     pitchData = parselSound.to_pitch_ac(time_step=stepSize/1000,
#                                         pitch_ceiling=400.0)
#     pitchValues = np.array(pitchData.selected_array['frequency'])

#     # Used for finding time stamp.
#     times = np.arange(0, len(data)/sampleRate, stepSizeInSamples/sampleRate) # seconds

#     # Step through each data point in formant and energy arrays and check for
#     # filler utterances over the next 'minimumLength' size window of features.
#     for step in range(0, numberOfSteps-utteranceWindowSize):

#         previousFilledPause = 0
#         if len(timeStamps) > 0:
#             previousFilledPause = timeStamps[-1]
#         else:
#             previousFilledPause = -10

#     if __debug__:
#         if DEBUG_FILLED_PAUSE_GRAPH:
#             # Show graph if debugging
#             filledPausesMarkers = np.full(len(timeStamps), 0)
#             energyTimes = np.arange(0, len(data)/sampleRate, stepSize/1000)[:len(energy)]

#             plt.figure(figsize=[16, 8])
#             plt.plot(energyTimes, energy, energyTimes, pitch)
#             plt.plot(np.array(timeStamps), filledPausesMarkers, 'go')
#             plt.show()
#             plt.close()

#     return filledPauses, np.array(timeStamps)

# # | [WIP] The start of an implementation of "A Simple but Efficient...".
# def getVoiceActivityMoattarAndHomayounpour(data, sampleRate):
#     # Primary thresholds
#     energyPrimaryThreshold = 40
#     frequencyPrimaryThreshold = 185
#     spectralFlatnessMeasurePrimaryThreshold = 5

#     frameSize = 10 # Milliseconds
#     numberOfFrames = len(data) / sampleRate * 1000

#     # Convert window and step sizes to samples for Librosa.
#     frameSizeInSamples = int(sampleRate / 1000 * frameSize)

#     dataFrames = librosa.util.frame(data, frame_length=frameSizeInSamples, hop_length=frameSizeInSamples)

#     voiceActivity = []

#     energies = []
#     dominantFrequencies = []
#     spectralFlatnessMeasures = []

#     silenceCount = 0

#     for frame in dataFrames:
#         currentActivity = 0

#         if counter > 1:
#             voiceActivity.append(1)
#         else:
#             voiceActivity.append(0)
#             silenceCount += 1
#             minEnergy = ( (silenceCount * minEnergy) + energy[i] ) / ( silenceCount + 1 )

#         energyThreshold = energyPrimaryThreshold * math.log(minEnergy)

#     return voiceActivity
