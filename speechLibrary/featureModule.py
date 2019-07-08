#
# Created on Apr 23, 2019
#
# @author: Julian Fortune
# @Description: Functions for extracting features from audio data.
#

import math
import numpy as np
import librosa
import scipy.signal # Extracts power spectrum
import parselmouth # Formants

import matplotlib.pyplot as plt

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

    def appendAllZeros(self):
        self.syllablesPerSecond = np.append(self.syllablesPerSecond, 0)
        self.meanVoiceActivity = np.append(self.meanVoiceActivity, 0)
        self.stDevVoiceActivity = np.append(self.stDevVoiceActivity, 0)
        self.meanPitch = np.append(self.meanPitch, 0)
        self.stDevPitch = np.append(self.stDevPitch, 0)
        self.meanIntensity = np.append(self.meanIntensity, 0)
        self.stDevIntensity = np.append(self.stDevIntensity, 0)

    def append(self, secondFeatureSet):
        self.syllablesPerSecond = np.append(self.syllablesPerSecond, secondFeatureSet.syllablesPerSecond)
        self.meanVoiceActivity = np.append(self.meanVoiceActivity, secondFeatureSet.meanVoiceActivity)
        self.stDevVoiceActivity = np.append(self.stDevVoiceActivity, secondFeatureSet.stDevVoiceActivity)
        self.meanPitch = np.append(self.meanPitch, secondFeatureSet.meanPitch)
        self.stDevPitch = np.append(self.stDevPitch, secondFeatureSet.stDevPitch)
        self.meanIntensity = np.append(self.meanIntensity, secondFeatureSet.meanIntensity)
        self.stDevIntensity = np.append(self.stDevIntensity, secondFeatureSet.stDevIntensity)


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
# | any of them are above the threshold
def aboveThresholdWithinTolerance(data, indexInQuestion, threshold, tolerance):
    window = tolerance * 2 - 1
    for index in range(indexInQuestion - tolerance,indexInQuestion + tolerance):
        if index >= 0 and index < len(data):
            if data[index] > threshold:
                return True
    return False

def getEnergyMinimumThreshold(energy):
    return np.percentile(energy, 10) * 2

def getEnergy(data, sampleRate, windowSize, stepSize):
    windowSizeInSamples = int(sampleRate / 1000 * windowSize)
    stepSizeInSamples = int(sampleRate / 1000 * stepSize)

    energy = librosa.feature.rmse(data, frame_length=windowSizeInSamples, hop_length=stepSizeInSamples)[0]
    return energy

def getShortTermEnergy(data, sampleRate, windowSize, stepSize):
    windowSizeInSamples = int(sampleRate / 1000 * windowSize)
    stepSizeInSamples = int(sampleRate / 1000 * stepSize)

    shortTermEnergy = np.array([
        sum(data[step:step+windowSizeInSamples]**2)
        for step in range(0, len(data), stepSizeInSamples)
    ])

    return shortTermEnergy

def getRMSPower(data, sampleRate, windowSize, stepSize):
    windowSizeInSamples = int(sampleRate / 1000 * windowSize)
    stepSizeInSamples = int(sampleRate / 1000 * stepSize)

    rms = np.array([
        math.sqrt(sum(data[step:step+windowSizeInSamples]**2) / windowSizeInSamples)
        for step in range(0, len(data), stepSizeInSamples)
    ])

    return rms

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

# | Gets power of sound and returns np arrays
def getPowerSpectrum(data,sampleRate,windowSize):
    freqs,ps = scipy.signal.welch(data,
                                  sampleRate,
                                  window='hanning',   # Apply a Hanning window
                                  nperseg= int(windowSize), # Compute periodograms of ____-long segments of x
                                  )
    return freqs,ps

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
def getSyllables(data, sampleRate, windowSize, stepSize, pitchValues, energyPeakMinimumDistance, energyPeakMinimumWidth, pitchDistanceTolerance, zcrThreshold):
    # Convert window and step sizes to samples for Librosa.
    windowSizeInSamples = int(sampleRate / 1000 * windowSize)
    stepSizeInSamples = int(sampleRate / 1000 * stepSize)

    # Get energy.
    energy = getEnergy(data, sampleRate, windowSize, stepSize)

    # Get energy threshold
    energyMinThreshold = getEnergyMinimumThreshold(energy)
    # Adjust energy threshold for pitch algorithm.
    fractionEnergyMinThreshold = energyMinThreshold / max(energy)

    # Get zero-crossing Rate
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=windowSizeInSamples, hop_length=stepSizeInSamples)[0]

    # Identify peaks in energy.
    peaks, _ = scipy.signal.find_peaks(energy,
                                       height=energyMinThreshold,
                                       distance=energyPeakMinimumDistance,
                                       width=energyPeakMinimumWidth)

    validPeaks = []

    # Remove candidate peaks that don't meet voicing requirements.
    for i in range(0,len(peaks)):
        if zcr[peaks[i]] < zcrThreshold and aboveThresholdWithinTolerance(data=pitchValues,
                                                                          indexInQuestion=peaks[i],
                                                                          threshold=0,
                                                                          tolerance=pitchDistanceTolerance):
            validPeaks = np.append(validPeaks, peaks[i])

    # Return syllables & candidate peaks that didn't meet voicing requirements.
    return np.array(validPeaks) * stepSizeInSamples / sampleRate, np.array(peaks) * stepSizeInSamples / sampleRate

# | Returns the voice activity (each v_i in V ∈ {0,1}) using an adaptive algorithm from "A Simple but Efficient...".
def getVoiceActivity(data, sampleRate, windowSizeInMS, stepSizeInMS, useAdaptiveThresholds, zcrThreshold, energyPrimaryThreshold, dominantFreqThreshold, dominantFreqTolerance):
    # Convert window and step sizes to samples for Librosa.
    windowSize = int(sampleRate / 1000 * windowSizeInMS) # samples
    stepSize = int(sampleRate / 1000 * stepSizeInMS) # samples

    dominantFrequency = []

    # Get most dominant frequency values.
    for i in range(math.ceil(len(data)/stepSize)):
        start = i*stepSize
        end = start+windowSize
        currentWindowSize = windowSize

        if end > len(data):
            end = len(data)
            currentWindowSize = len(data[start:end])
        freq, ps = getPowerSpectrum(data[start:end],sampleRate,currentWindowSize)
        dominantFrequency.append(freq[np.argmax(ps)])

    # Get energy and zero-crossing rate.
    energy = getShortTermEnergy(data, sampleRate, windowSizeInMS, stepSizeInMS)
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=windowSize, hop_length=stepSize)[0]

    # Ensure the arrays all line up correctly.
    assert (len(zcr) == len(energy) and len(energy) == len(dominantFrequency)), "Feature lengths differ."

    # Move the thresholds if needed.
    if useAdaptiveThresholds:
        minEnergy = np.mean(energy[0:30])
        energyThreshold = energyPrimaryThreshold * math.log(minEnergy)
    else:
        energyThreshold = energyPrimaryThreshold

    voiceActivity = []
    silenceCount = 0

    for i in range(0,len(energy)):
        currentActivity = 0

        if  zcr[i] < zcrThreshold and energy[i] > energyThreshold and aboveThresholdWithinTolerance(dominantFrequency,
                                                                                                    i,
                                                                                                    dominantFreqThreshold,
                                                                                                    dominantFreqTolerance):

            currentActivity = 1 # Voice acitivty present
        else:
            silenceCount += 1

        voiceActivity = np.append(voiceActivity, currentActivity) # No voice acitivty present

        if useAdaptiveThresholds:
            minEnergy = ( (silenceCount * minEnergy) + energy[i] ) / ( silenceCount + 1 )
            energyThreshold = energyPrimaryThreshold * math.log(minEnergy)

    return voiceActivity

# | Returns an array of timestamps where filled pauses were detected.
def getFilledPauses(data, sampleRate, windowSize, stepSize, minumumLength, F1MaximumVariance, F2MaximumVariance, energyThreshold):
    # Convert window and step sizes to samples for Librosa and to prevent rounding issues with RMSE.
    sampleWindowSize = int(windowSize*sampleRate/1000)
    sampleStepSize = int(stepSize*sampleRate/1000)

    timeStamps = []
    lengths = []

    # Get first and second formants (F1 & F2) and energy.
    firstFormant, secondFormant = getFormants(data, sampleRate, sampleWindowSize/sampleRate, sampleStepSize/sampleRate)
    energy = getEnergy(data, sampleRate, windowSize, stepSize)

    harmonicity = librosa.feature.spectral_flatness(data, hop_length=sampleStepSize)[0][:len(energy)]

    energyThreshold = getEnergyMinimumThreshold(energy)

    # The number of steps in energy and formant arrays
    numberOfSteps = round((len(data)/sampleRate) / (sampleStepSize/sampleRate))

    # Used for finding time stamp.
    times = np.arange(0, len(data)/sampleRate, sampleStepSize/sampleRate) # seconds

    # The number of steps in the feature arrays that make up a single window for checking for utterances.
    utteranceWindowSize = int(minumumLength / 1000 * sampleRate / sampleStepSize)

    fillerUtteranceInitiated = False
    startOfFiller = 0

    # Step through each data point in formant and energy arrays and check for
    # filler utterances over the next 'minimumLength' size window of features.
    for step in range(0, numberOfSteps-utteranceWindowSize):
        start = step
        end = step+utteranceWindowSize

        firstFormantVariance = np.std(firstFormant[start:end])
        secondFormantVariance = np.std(secondFormant[start:end])
        averageFormantDistance = np.mean(secondFormant[start:end] - firstFormant[start:end])

        averageEnergy = np.mean(energy[start:end])
        harmonicityVariance = np.std(harmonicity[start:end])

        # Check for any filled pauses immediately before the current window
        previousFilledPause = 0
        if len(timeStamps) > 0:
            previousFilledPause = timeStamps[-1]
        else:
            previousFilledPause = -10
        distanceToPreviousFilledPause = times[step] - previousFilledPause

        # Identify filled pauses
        if firstFormantVariance <= F1MaximumVariance and secondFormantVariance <= F2MaximumVariance and averageEnergy > energyThreshold and distanceToPreviousFilledPause > 1 and averageFormantDistance < 2000 and harmonicityVariance < 0.001:
            # Prevent an utterance from being detected many times.
            if fillerUtteranceInitiated == False:
                fillerUtteranceInitiated = True

                timeStamps.append(times[step])
                startOfFiller = times[step]
        else:
            # Keep track of the length of the filled pauses.
            if fillerUtteranceInitiated == True:
                lengths.append(times[step] - startOfFiller + (minumumLength/1000) )

            fillerUtteranceInitiated = False

    return np.array(timeStamps)

# | Computes the absolute value of the raw data values then calculates
# | the mean, max, min, and standard deviation of those data values
def getIntensityStatistics(data):
    absVal = np.absolute(data)
    average = np.mean(absVal)
    stDev = np.std(absVal)
    return average, stDev

# | Computes welch looking back and for number of sampleRate in length of data
# | and returns the average of the loudest pitch in each window
def getPitchStatistics(data, sampleRate, windowSize):

    pitches = getPitch(data, sampleRate, stepSize, silenceProportionThreshold)

    average = np.mean(loudestPitch)
    stDev = np.std(loudestPitch)

    return average, stDev

# | Returns the average voice activity (0 <= v <= 1) and stDev using the getVoiceActivity() method.
def getVoiceActivityStatistics(data, sampleRate, windowSizeInMS, stepSizeInMS, useAdaptiveThresholds, zcrThreshold, energyPrimaryThreshold, dominantFreqThreshold, dominantFreqTolerance):
    voiceActivity = getVoiceActivity(data, sampleRate, windowSizeInMS, stepSizeInMS, useAdaptiveThresholds, zcrThreshold, energyPrimaryThreshold, dominantFreqThreshold, dominantFreqTolerance)

    # Get stats on voice activity
    average = np.mean(voiceActivity)
    stDev = np.std(voiceActivity)
    return average, stDev
