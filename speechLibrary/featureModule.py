'''
Created on Apr 23, 2019

@author: Julian Fortune

@Description: Functions for extracting features from audio data.
'''
import math
import numpy
import librosa
import scipy.signal # Extracts power spectrum

class FeatureSet:

    # TODO: Redo to use classic lists and convert at very end

    def __init__(self):
        self.syllablesPerSecond = numpy.zeros(shape=0)
        self.meanVoiceActivity = numpy.zeros(shape=0)
        self.stDevVoiceActivity = numpy.zeros(shape=0)
        self.meanPitch = numpy.zeros(shape=0)
        self.stDevPitch = numpy.zeros(shape=0)
        self.meanIntensity = numpy.zeros(shape=0)
        self.stDevIntensity = numpy.zeros(shape=0)

    def appendAllZeros(self):
        self.syllablesPerSecond = numpy.append(self.syllablesPerSecond, 0)
        self.meanVoiceActivity = numpy.append(self.meanVoiceActivity, 0)
        self.stDevVoiceActivity = numpy.append(self.stDevVoiceActivity, 0)
        self.meanPitch = numpy.append(self.meanPitch, 0)
        self.stDevPitch = numpy.append(self.stDevPitch, 0)
        self.meanIntensity = numpy.append(self.meanIntensity, 0)
        self.stDevIntensity = numpy.append(self.stDevIntensity, 0)

    def append(self, secondFeatureSet):
        self.syllablesPerSecond = numpy.append(self.syllablesPerSecond, secondFeatureSet.syllablesPerSecond)
        self.meanVoiceActivity = numpy.append(self.meanVoiceActivity, secondFeatureSet.meanVoiceActivity)
        self.stDevVoiceActivity = numpy.append(self.stDevVoiceActivity, secondFeatureSet.stDevVoiceActivity)
        self.meanPitch = numpy.append(self.meanPitch, secondFeatureSet.meanPitch)
        self.stDevPitch = numpy.append(self.stDevPitch, secondFeatureSet.stDevPitch)
        self.meanIntensity = numpy.append(self.meanIntensity, secondFeatureSet.meanIntensity)
        self.stDevIntensity = numpy.append(self.stDevIntensity, secondFeatureSet.stDevIntensity)


# | Gets power of sound and returns numpy arrays
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
def getSyllables(data, sampleRate, frameSize, hopSize, peakMinDistance, peakMinWidth, zcrThreshold, dominantFreqThreshold, dominantFreqTolerance):
    ### Constants
    frame = int(sampleRate / 1000 * frameSize) # samples
    hop = int(sampleRate / 1000 * hopSize) # samples

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
        dominantFrequency.append(freq[numpy.argmax(ps)])

    ### Energy
    energy = librosa.feature.rmse(data, frame_length=frame, hop_length=hop)[0]

    ### Threshold
    energyMinThreshold = numpy.median(energy) * 2

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
            validPeaks = numpy.append(validPeaks, peaks[i])

    return validPeaks

# |
def getVoiceActivityFromSyllables(start,end,syllables,sampleRate,hop):
    for index in syllables:
        sampleIndex = index * sampleRate / 1000 * 16
        if start <= sampleIndex and sampleIndex < end:
            return 1
    return 0

# | Returns the average voice activity (0 <= v <= 1)
def getVoiceActivityFeatures(length,sampleRate,windowSize,syllables,hop):
    sampleWindowSize = windowSize * sampleRate
    voiceActivity = numpy.zeros(shape=0)

    step = 0
    while step < length:
        voiceActivity = numpy.append(voiceActivity,getVoiceActivityFromSyllables(step, step + sampleWindowSize,syllables,sampleRate,hop))

        # Increment to next step
        step += sampleWindowSize

    averageVoiceActivity = numpy.mean(voiceActivity)
    voiceActivityStDev = numpy.std(voiceActivity)
    return averageVoiceActivity, voiceActivityStDev

def adaptiveVAD(data, sampleRate, frameSize, hopSize, medianInitialThresholds, adaptiveThresholds):
        ### Constants
        frame = int(sampleRate / 1000 * frameSize) # samples
        hop = int(sampleRate / 1000 * hopSize) # samples

        ### Dominant frequency analysis
        dominantFrequency = []
        windowSize = frame

        for i in range(int(len(data)/hop) + 1): #sketchy but works; TODO fix
            start = i*hop
            end = start+frame
            if end > len(data):
                end = len(data)
                windowSize = len(data[start:end])
            freq, ps = getPowerSpectrum(data[start:end],sampleRate,windowSize)
            dominantFrequency.append(freq[numpy.argmax(ps)])

        ### Energy
        energy = librosa.feature.rmse(data, frame_length=frame, hop_length=hop)[0]

        ### ZCR
        zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame, hop_length=hop)[0]

        ###Thresholds
        dominantFreqTolerance = 8

        if medianInitialThresholds == True:
            zcrThreshold = numpy.median(zcr)
            energyPrimaryThreshold = numpy.median(energy)
            dominantFreqThreshold = numpy.median(dominantFrequency)
        else:
            zcrThreshold = 0.06
            energyPrimaryThreshold = 40
            dominantFreqThreshold = 18

        if adaptiveThresholds:
            minEnergy = numpy.mean(energy[0:30])
            energyThreshold = energyPrimaryThreshold * math.log(minEnergy)
        else:
            nergyThreshold = energyPrimaryThreshold

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

            voiceActivity = numpy.append(voiceActivity, currentActivity) # No voice acitivty present

            if adaptiveThresholds:
                minEnergy = ( (silenceCount * minEnergy) + energy(i) ) / ( silenceCount + 1 )
                energyThreshold = energyPrimaryThreshold * math.log(minEnergy)

        return voiceActivity

# | Computes the absolute value of the raw data values then calculates
# | the mean, max, min, and standard deviation of those data values
def getIntensityFeatures(data):
    absVal = numpy.absolute(data)
    average = numpy.mean(absVal)
    max = numpy.amax(absVal)
    min = numpy.amin(absVal)
    stDev = numpy.std(absVal)
    return average, stDev

# | Computes welch looking back and for number of sampleRate in length of data
# | and returns the average of the loudest pitch in each window
def getPitchFeatures(data,sampleRate,windowSize):
    sampleWindowSize = windowSize * sampleRate # windowSize in seconds
    loudestPitch = numpy.zeros(shape=0)

    step = 0
    while step < len(data):
        freqs, ps = getPowerSpectrum(data[step:step + sampleWindowSize],sampleRate,sampleRate)
        loudestPitch = numpy.append(loudestPitch, numpy.argmax(ps))

        # Increment to next step
        step += sampleWindowSize
    average = numpy.mean(loudestPitch)
    stDev = numpy.std(loudestPitch)

    return average, stDev
