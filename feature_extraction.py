import os, numpy, scipy, time, io, glob, librosa
import scipy.signal # Extracts power spectrum
import wavio # Converts wav to numpy array
import matplotlib.pyplot as plt # Visualisation
import math

numpy.set_printoptions(threshold=numpy.nan)

### Utilities

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

def simpleVAD(data, sampleRate, frameSize, hopSize, medianInitialThresholds, adaptiveThresholds):
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

# | Extracts all features and returns array in accordance with Jamison's drawing
# | Parameters:
# |   - filePath: path to file to process
# |   - stepSize: size of step in seconds
# |   - lookBackSize: size of window for looking back on longterm features in seconds
# |   - percentageThreshold: percentage of loudest noise in band above which vocal activity is present
# |   - printStatus: if True will print out as each second is processed, else no print
# | Returns:
# |   - Numpy array with features
def getFeatures(filePath,stepSize,lookBackSize,syllableUseMediansForThresholds,
    syllableAdaptive,percentageThreshold,printStatus):

    if printStatus :
        print("[ START ] Working on:",filePath)

    # Read in the file, extract data and metadata
    audioData = wavio.read(filePath) # reads in audio file
    numberOfChannels = len(audioData.data[0])
    sampleRate = audioData.rate # usually 44100
    width = audioData.sampwidth # bit depth is equal to width times 8
    length = len(audioData.data) # gets number of sample in audio

    # Make the audio data mono
    data = numpy.mean(audioData.data,axis=1)

    # Set up arrays for features
    seconds = numpy.zeros(shape=0)
    syllablesPerSecond = numpy.zeros(shape=0)
    meanVoiceActivity = numpy.zeros(shape=0)
    stDevVoiceActivity = numpy.zeros(shape=0)
    meanPitch = numpy.zeros(shape=0)
    stDevPitch = numpy.zeros(shape=0)
    meanIntensity = numpy.zeros(shape=0)
    stDevIntensity = numpy.zeros(shape=0)

    step = 0
    sampleStepSize = int(stepSize * sampleRate)
    sampleLookBackSize = int(lookBackSize * sampleRate)
    while step < length:
        if printStatus:
            print("[",str(step/length*100)[:4],"% ] Second",int(step/sampleRate))

        # keep track of what second we're in
        seconds = numpy.append(seconds,step/sampleRate)

        # look backward to calculate features over long term
        if step + sampleStepSize - sampleLookBackSize > 0:

            lookBackChunk = data[step + sampleStepSize - sampleLookBackSize:step + sampleStepSize]

            ### WORDS PER MINUTE
            frameSizeMS = 64
            hopSizeMS = 16
            peakMinDistance = 5
            peakMinWidth = 2
            zcrThreshold = 0.06
            dominantFreqThreshold = 200
            dominantFreqTolerance = 8

            syllables = getSyllables(lookBackChunk, sampleRate, frameSizeMS, hopSizeMS, peakMinDistance, peakMinWidth, zcrThreshold, dominantFreqThreshold, dominantFreqTolerance)
            currentSyllablesPerSecond = len(syllables)/lookBackSize
            syllablesPerSecond = numpy.append(syllablesPerSecond,currentSyllablesPerSecond)

            ### VAD
            average, stDev = getVoiceActivityFeatures(len(lookBackChunk),sampleRate,1,syllables,hopSizeMS)
            meanVoiceActivity = numpy.append(meanVoiceActivity,average)
            stDevVoiceActivity = numpy.append(stDevVoiceActivity,stDev)

            ### AVERAGE PITCH
            average, stDev = getPitchFeatures(lookBackChunk,sampleRate,1)
            meanPitch = numpy.append(meanPitch, average)
            stDevPitch = numpy.append(stDevPitch, stDev)

            ### INTENSITY FEATURES
            average, stDev = getIntensityFeatures(lookBackChunk)
            meanIntensity = numpy.append(meanIntensity, average)
            stDevIntensity = numpy.append(stDevIntensity, stDev)

        # Fills arrays with zeros until step is larger than lookBackSize
        else:
            syllablesPerSecond = numpy.append(syllablesPerSecond, 0)
            meanVoiceActivity = numpy.append(meanVoiceActivity, 0)
            stDevVoiceActivity = numpy.append(stDevVoiceActivity, 0)
            meanPitch = numpy.append(meanPitch, 0)
            stDevPitch = numpy.append(stDevPitch, 0)
            meanIntensity = numpy.append(meanIntensity, 0)
            stDevIntensity = numpy.append(stDevIntensity, 0)

        # Increment to next step
        step += sampleStepSize

    # Pulls all the feautures together in one array
    features = numpy.vstack([seconds,syllablesPerSecond,meanVoiceActivity,stDevVoiceActivity,meanPitch,stDevPitch,meanIntensity,stDevIntensity])

    if printStatus :
        print("[ DONE ] Finished processing",filePath,"!")

    return features

def getFeaturesOnAllFilesInDirectory():
    dir = "../media/Participant_Audio/*.wav"

    # Keep track of running stats
    startTime = time.time()
    count = 1

    for path in sorted(glob.iglob(dir),reverse=False):
        # Communicate progress
        print("[ " + str(count) + "/" + str(len(sorted(glob.iglob(dir)))) + " ] \tStarting:",path)

        featureArray = getFeatures(filePath=path, #file
                                   stepSize=1, #how big to step in seconds
                                   lookBackSize=30,  #how big of interval to wait until looking for transcript, pitch/intensity features in seconds
                                   syllableUseMediansForThresholds = True,
                                   syllableAdaptive = False,
                                   percentageThreshold=0.001,  #percentage of loudest noise in band above which vocal activity is present
                                   printStatus=True
                                   )

        # Save the numpy array
        numpy.save("./features/" + os.path.basename(path)[:-4],featureArray)

        # Crunch some numbers and communicate to the user
        timeElapsed = time.time() - startTime
        estimatedTimeRemaining = timeElapsed/count * (len(sorted(glob.iglob(dir))) - count)
        print("\t\t" + str(timeElapsed/60) + " minutes elapsed. Estimated time remaining: " + str(estimatedTimeRemaining/60))

        count += 1

def main():
    # getFeaturesOnAllFilesInDirectory()

    dir = "../media/Participant_Audio/*.wav"

    for filePath in sorted(glob.iglob(dir),reverse=False):

        # Read in the file, extract data and metadata
        audioData = wavio.read(filePath) # reads in audio file
        numberOfChannels = len(audioData.data[0])
        sampleRate = audioData.rate # usually 44100
        width = audioData.sampwidth # bit depth is equal to width times 8
        length = len(audioData.data) # gets number of sample in audio

        # Make the audio data mono
        data = numpy.mean(audioData.data,axis=1)

        print(simpleVAD(data, sampleRate, 64, 16, "firstFrames", False))


if __name__ == "__main__":
    main()
