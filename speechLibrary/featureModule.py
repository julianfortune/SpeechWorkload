'''
Created on Apr 23, 2019

@author: Julian Fortune

@Description: Functions for extracting features from audio data.
'''
import math
import numpy as np
import librosa
import scipy.signal # Extracts power spectrum
import parselmouth # Formants

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

# | Gets power of sound and returns np arrays
def getPowerSpectrum(data,sampleRate,windowSize):
    freqs,ps = scipy.signal.welch(data,
                                  sampleRate,
                                  window='hanning',   # Apply a Hanning window
                                  nperseg= int(windowSize), # Compute periodograms of ____-long segments of x
                                  )
    return freqs,ps

# | Checks surrounding values in an array around the index to check if
# | any of them are above the threshold
def aboveThresholdWithinTolerance(data,indexInQuestion,threshold,tolerance):
    window = tolerance * 2 - 1
    for index in range(indexInQuestion - tolerance,indexInQuestion + tolerance):
        if index >= 0 and index < len(data):
            if data[index] > threshold:
                return True
    return False

# | Returns the indices of syllables in audio data
def getSyllables(data, sampleRate, windowSize, stepSize, peakMinDistance, peakMinWidth, zcrThreshold, dominantFreqThreshold, dominantFreqTolerance):
    ### Constants
    frame = int(sampleRate / 1000 * windowSize) # samples
    hop = int(sampleRate / 1000 * stepSize) # samples

    ### Dominant frequency analysis
    dominantFrequency = []
    windowSize = frame

    for i in range(int(len(data)/hop)):
        start = i*hop
        end = start+frame
        if end > len(data):
            end = len(data)
            windowSize = len(data[start:end])
        freq, ps = getPowerSpectrum(data[start:end],sampleRate,windowSize)
        dominantFrequency.append(freq[np.argmax(ps)])

    ### Energy
    energy = librosa.feature.rmse(data, frame_length=frame, hop_length=hop)[0]

    ### Threshold
    energyMinThreshold = np.median(energy) * 2

    ### Peaks
    peaks, _ = scipy.signal.find_peaks(energy,
                                       height=energyMinThreshold,
                                       distance=peakMinDistance,
                                       width=peakMinWidth)

    ### ZCR
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame, hop_length=hop)[0]

    ### Removing invalid peaks
    validPeaks = []
    for i in range(0,len(peaks)):
        if zcr[peaks[i]] < zcrThreshold and aboveThresholdWithinTolerance(dominantFrequency,
                                                                          peaks[i],
                                                                          dominantFreqThreshold,
                                                                          dominantFreqTolerance):
            validPeaks = np.append(validPeaks, peaks[i])

    return validPeaks

# | Returns the average voice activity (0 <= v <= 1) using an adaptive algorithm.
def getVoiceActivity(data, sampleRate, windowSizeInMS, stepSizeInMS, useAdaptiveThresholds, zcrThreshold, energyPrimaryThreshold, dominantFreqThreshold, dominantFreqTolerance):
    ### Constants
    windowSize = int(sampleRate / 1000 * windowSizeInMS) # samples
    stepSize = int(sampleRate / 1000 * stepSizeInMS) # samples

    ### Dominant frequency analysis
    dominantFrequency = []
    currentWindowSize = windowSize

    for i in range(math.ceil(len(data)/stepSize)):
        start = i*stepSize
        end = start+windowSize

        if end > len(data):
            end = len(data)
            currentWindowSize = len(data[start:end])
        freq, ps = getPowerSpectrum(data[start:end],sampleRate,currentWindowSize)
        dominantFrequency.append(freq[np.argmax(ps)])

    ### Energy
    energy = librosa.feature.rmse(data, frame_length=windowSize, hop_length=stepSize)[0]

    ### ZCR
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=windowSize, hop_length=stepSize)[0]

    if useAdaptiveThresholds:
        minEnergy = np.mean(energy[0:30])
        energyThreshold = energyPrimaryThreshold * math.log(minEnergy)
    else:
        energyThreshold = energyPrimaryThreshold

    ### Going through each frame
    voiceActivity = []
    silenceCount = 0

    if len(zcr) != len(energy) or len(energy) != len(dominantFrequency):
        print("Error! Lengths differ!")
        print("zcr: ", len(zcr), " energy: ", len(energy), " dominantFreq: ", len(dominantFrequency))
        return voiceActivity

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

# | Returns the average voice activity (0 <= v <= 1) using an adaptive algorithm.
def getVoiceActivityStatistics(data, sampleRate, windowSizeInMS, stepSizeInMS, useAdaptiveThresholds, zcrThreshold, energyPrimaryThreshold, dominantFreqThreshold, dominantFreqTolerance):
    # Voice activity
    voiceActivity = getVoiceActivity(data, sampleRate, windowSizeInMS, stepSizeInMS, useAdaptiveThresholds, zcrThreshold, energyPrimaryThreshold, dominantFreqThreshold, dominantFreqTolerance)

    # Get stats on voice activity
    average = np.mean(voiceActivity)
    stDev = np.std(voiceActivity)
    return average, stDev

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
    sampleWindowSize = int(windowSize / 1000 * sampleRate)
    loudestPitch = np.zeros(shape=0)

    step = 0
    while step < len(data):
        freqs, ps = getPowerSpectrum(data[step:step + sampleWindowSize],sampleRate,sampleRate)
        loudestPitch = np.append(loudestPitch, np.argmax(ps))

        # Increment to next step
        step += sampleWindowSize
    average = np.mean(loudestPitch)
    stDev = np.std(loudestPitch)

    return average, stDev

# | Returns the first two formant from a piece of audio.
def getFormants(data, sampleRate, windowSize, stepSize):
    # Convert to the parselmouth custom sound type (req'd for formant function)
    parselSound = parselmouth.Sound(values=data, sampling_frequency=sampleRate)

    # Produce a formant object from this data
    formantData = parselSound.to_formant_burg(window_length=windowSize, time_step=stepSize)

    # API for parselmouth doesn't explain how to extract the formatData without querying for everything manually.
    firstFormant = []
    secondFormant = []

    # Used for plots
    times = np.arange(0, len(data)/sampleRate, stepSize) # seconds

    for timeStamp in times:
        firstFormantValue = formantData.get_value_at_time(1, timeStamp)
        secondFormantValue = formantData.get_value_at_time(2, timeStamp)

        firstFormant.append(firstFormantValue)
        secondFormant.append(secondFormantValue)

    firstFormant = np.array(firstFormant)
    secondFormant = np.array(secondFormant)

    return firstFormant, secondFormant

# | Returns a np array for each frame with 1 for filled pause, 0 for no filled pause
# | and an array of timesstamps where filled pauses were detected
def getFilledPauses(data, sampleRate, windowSize, stepSize, minumumLength, F1MaximumVariance, F2MaximumVariance, energyThreshold):
    # Everything needs to be based on samples to prevent rounding issues with RMSE
    sampleWindowSize = int(windowSize*sampleRate/1000)
    sampleStepSize = int(stepSize*sampleRate/1000)

    # The number of steps in energy and formant arrays
    numberOfSteps = round((len(data)/sampleRate) / (sampleStepSize/sampleRate))

    # Formant extraction
    firstFormant, secondFormant = getFormants(data, sampleRate, sampleWindowSize/sampleRate, sampleStepSize/sampleRate)

    # Energy
    energy = librosa.feature.rmse(data, frame_length=sampleWindowSize, hop_length=sampleStepSize)[0]

    # Filled pauses detection
    filledPauses = np.zeros(numberOfSteps)
    timeStamps = []
    lengths = []

    # Visualization helpers
    firstFormantVariances = []
    secondFormantVariances = []
    averageEnergies = []
    stepTimes = []

    # Used for plots
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
        averageEnergy = np.mean(energy[start:end])

        stepTimes.append(start * (sampleStepSize/sampleRate))
        firstFormantVariances.append(firstFormantVariance)
        secondFormantVariances.append(secondFormantVariance)
        averageEnergies.append(averageEnergy)

        if firstFormantVariance <= F1MaximumVariance and secondFormantVariance <= F2MaximumVariance and averageEnergy > energyThreshold:
            # Prevent an utterance from being detected many times
            if fillerUtteranceInitiated == False:
                filledPauses[step] = 1
                fillerUtteranceInitiated = True
                timeStamps.append(times[step])

                # print("  ", times[step], firstFormantVariance, secondFormantVariance, averageEnergy)

                startOfFiller = times[step]
        else:
            if fillerUtteranceInitiated == True:
                lengths.append(times[step] - startOfFiller + (minumumLength/1000) )
            fillerUtteranceInitiated = False

    return filledPauses, np.array(timeStamps), times, firstFormant, secondFormant, energy, lengths, firstFormantVariances, secondFormantVariances, averageEnergies, stepTimes
