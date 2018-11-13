import os, numpy, scipy, time, io, glob
import scipy.signal # Extracts power spectrum
from scipy.io.wavfile import read, write # Writes temp file of window for sr to use
import wavio # Converts wav to numpy array
import speech_recognition as sr # Sphynx transcript
import matplotlib.pyplot as plt # Visualisation

numpy.set_printoptions(threshold=numpy.inf)

# | Gets transcript of audio converted to text by Sphinx, returns array of
# | words (as strings). Writes "<fileName>-temp.wav" to work with sr; deletes before exiting
# | Returns:
# |   Python array of words detected in audio
def getTranscriptFromData(data,sampleRate,filePath):
    tempFilePath = filePath + "-temp.wav"
    wavio.write(tempFilePath, data, sampleRate, sampwidth=1)

    r = sr.Recognizer() # Initialize recognizer object
    r.energy_threshold = 200;
    with sr.AudioFile(tempFilePath) as source:
        audio = r.record(source)  # Reads in the entire audio file
    transcript = []
    try:
        transcript += r.recognize_sphinx(audio).split()
    except:
        print("Sphynx encountered an error!")

    os.remove(tempFilePath)

    return transcript

# | Gets power of sound and returns numpy arrays
def getPowerSpectrum(data,sampleRate,windowSize):
    freqs,ps = scipy.signal.welch(data,
                                  sampleRate,
                                  window='hanning',   # Apply a Hanning window
                                  nperseg= int(windowSize), # Compute periodograms of ____-long segments of x
                                  )
    return freqs,ps

# | Compares power of each window to THRESHOLD and returns numpy array with
# | 1s representing speech and 0s representing silence
def getVoiceActivity(data,sampleRate,thresholdForSpeech):
    minFreq = 300 # In hertz the lowest frequency of the voice band
    maxFreq = 3000 # In hertz the highest frequency of the voice band

    freqs, ps = getPowerSpectrum(data,sampleRate,len(data))

    if numpy.mean(ps[minFreq:maxFreq]) > thresholdForSpeech:
        return 1
    else:
        return 0
    return 0

# | Returns the average voice activity (0 <= v <= 1)
def getVoiceActivityFeatures(data,sampleRate,windowSize,thresholdForSpeech):
    sampleWindowSize = windowSize * sampleRate
    voiceActivity = numpy.zeros(shape=0)

    step = 0
    while step < len(data):
        voiceActivity = numpy.append(voiceActivity,getVoiceActivity(data[step:step + sampleWindowSize],sampleRate,thresholdForSpeech))

        # Increment to next step
        step += sampleWindowSize

    averageVoiceActivity = numpy.mean(voiceActivity)
    voiceActivityStDev = numpy.std(voiceActivity)
    return averageVoiceActivity, voiceActivityStDev


# | Goes over whole audiofile and finds loudest noise in band of vocal activity,
# | multiplies by the supplied percentage and returns. Any noise above this
# | level in the voice band will be identified as speech
def getVADThreshold(data,sampleRate,minFreq,maxFreq,percentageThreshold,windowSize=1):
    sampleWindowSize = windowSize * sampleRate # #windowSize in seconds converted to samples

    rms = numpy.empty((0))

    for window in range(0,int(len(data) / sampleWindowSize)):
        windowStart = int(window * sampleWindowSize)
        windowEnd = int(windowStart + sampleWindowSize)
        if windowEnd < len(data): # Catch running off the end of the wav array
            freqs,ps = scipy.signal.welch(data[windowStart:windowEnd],
                                          sampleRate,
                                          window='hanning',   # Apply a Hanning window
                                          nperseg= int(sampleWindowSize), # Compute periodograms of ____-long segments of x
                                          )

            rms = numpy.append(rms, numpy.mean(ps[minFreq:maxFreq]))

    threshold = numpy.amax(rms) * percentageThreshold

    return threshold

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
def getFeatures(filePath,stepSize,lookBackSize,percentageThreshold,printStatus):
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
    wpm = numpy.zeros(shape=0)
    meanVoiceActivity = numpy.zeros(shape=0)
    stDevVoiceActivity = numpy.zeros(shape=0)
    meanPitch = numpy.zeros(shape=0)
    stDevPitch = numpy.zeros(shape=0)
    meanIntensity = numpy.zeros(shape=0)
    stDevIntensity = numpy.zeros(shape=0)

    if printStatus :
        print("Determining threshold for speech...")

    threshold = getVADThreshold(data,           # audio data
                                sampleRate=sampleRate,
                                minFreq=300,    # minimum frequency of vocal activity
                                maxFreq=3000,   # max frequency of vocal activity
                                percentageThreshold=percentageThreshold
                                )

    if printStatus :
        print("Done!")

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
            transcript = getTranscriptFromData(lookBackChunk,sampleRate,filePath)
            currentWPM = len(transcript)/( lookBackSize/60 )
            wpm = numpy.append(wpm,currentWPM)

            ### VAD
            average, stDev = getVoiceActivityFeatures(lookBackChunk,sampleRate,1,threshold)
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
            wpm = numpy.append(wpm, 0)
            meanVoiceActivity = numpy.append(meanVoiceActivity, 0)
            stDevVoiceActivity = numpy.append(stDevVoiceActivity, 0)
            meanPitch = numpy.append(meanPitch, 0)
            stDevPitch = numpy.append(stDevPitch, 0)
            meanIntensity = numpy.append(meanIntensity, 0)
            stDevIntensity = numpy.append(stDevIntensity, 0)

        # Increment to next step
        step += sampleStepSize

    # Pulls all the feautures together in one array
    features = numpy.vstack([seconds,wpm,meanVoiceActivity,stDevVoiceActivity,meanPitch,stDevPitch,meanIntensity,stDevIntensity])

    if printStatus :
        print("[ DONE ] Finished processing",filePath,"!")

    return features

def main():
    dir = "../media/Participant_Audio/*.wav"

    # Keep track of running stats
    startTime = time.time()
    count = 1

    for path in sorted(glob.iglob(dir)):
        # Communicate progress
        print("[ " + str(count) + "/" + str(len(sorted(glob.iglob(dir)))) + " ] \tStarting:",path)

        featureArray = getFeatures(filePath=path, #file
                                   stepSize=1, #how big to step in seconds
                                   lookBackSize=30,  #how big of interval to wait until looking for transcript, pitch/intensity features in seconds
                                   percentageThreshold=0.001,  #percentage of loudest noise in band above which vocal activity is present
                                   printStatus=False
                                   )

        # Save the numpy array
        numpy.save("./features/" + os.path.basename(path)[:-4],featureArray)

        # Crunch some numbers and communicate to the user
        timeElapsed = time.time() - startTime
        estimatedTimeRemaining = timeElapsed/count * (len(sorted(glob.iglob(dir))) - count)
        print("\t\t" + str(timeElapsed/60) + " minutes elapsed. Estimated time remaining: " + str(estimatedTimeRemaining/60))

        count += 1

if __name__ == "__main__":
    main()
