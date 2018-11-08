import os, numpy, scipy, time, io
import scipy.signal # used to extract power spectrum
from scipy.io.wavfile import read, write #writes temp file of window for sr to use
import wavio # converts wav to numpy array
import speech_recognition as sr # neede for sphync transcript
import matplotlib.pyplot as plt # needed for visualisation

numpy.set_printoptions(threshold=numpy.inf)

# | Gets transcript of audio converted to text by Sphinx, returns array of
# | words (strings). Writes "<fileName>-temp.wav" to work with sr; deletes before exiting
def getTranscriptFromData(data,sampleRate,filePath):

    tempFilePath = filePath + "-temp.wav"

    wavio.write(tempFilePath, data, sampleRate, sampwidth=1)

    r = sr.Recognizer() # initialize recognizer object
    r.energy_threshold = 200;
    with sr.AudioFile(tempFilePath) as source:
        audio = r.record(source)  # reads in the entire audio file
    transcript = []
    try:
        transcript += r.recognize_sphinx(audio).split()
    except:
        print("Sphynx encountered an error!")

    os.remove(tempFilePath)

    return transcript #returns python array of words detected in audio

# | Gets power of sound and returns numpy arrays
def getPowerSpectrum(data,sampleRate,windowSize):
    freqs,ps = scipy.signal.welch(data,
                                  sampleRate,
                                  window='hanning',   # apply a Hanning window
                                  nperseg= int(windowSize), # compute periodograms of ____-long segments of x
                                  )
    return freqs,ps

# | Compares power of each window to THRESHOLD and returns numpy array with
# | 1s representing speech and 0s representing silence
def getVoiceActivity(data,sampleRate,windowSize,thresholdForSpeech):
    minFreq = 300 # in hertz the lowest frequency of the voice band
    maxFreq = 3000 # in hertz the highest frequency of the voice band

    if len(data) < windowSize:
        windowSize = len(data)

    freqs, ps = getPowerSpectrum(data,sampleRate,windowSize)

    if numpy.mean(ps[minFreq:maxFreq]) > thresholdForSpeech:
        return 1
    else:
        return 0
    return 0

# | Returns the average voice activity (0 <= v <= 1)
def getAverageVoiceActivity():
    voiceActivity = numpy.empty

    numpy.append(voiceActivity,getVoiceActivity(chunk,sampleRate,sampleStepSize,threshold))

# | Goes over whole audiofile and finds loudest noise in band of vocal activity,
# | multiplies by the supplied percentage and returns. Any noise above this
# | level in the voice band will be identified as speech
def getVADThreshold(data,sampleRate,minFreq,maxFreq,percentageThreshold,windowSize=1):
    sampleWindowSize = windowSize * sampleRate # #windowSize in seconds converted to samples

    rms = numpy.empty((0))

    for window in range(0,int(len(data) / sampleWindowSize)):
        windowStart = int(window * sampleWindowSize)
        windowEnd = int(windowStart + sampleWindowSize)
        if windowEnd < len(data): #catch running off the end of the wav array
            freqs,ps = scipy.signal.welch(data[windowStart:windowEnd],
                                          sampleRate,
                                          window='hanning',   # apply a Hanning window
                                          nperseg= int(sampleWindowSize), # compute periodograms of ____-long segments of x
                                          )

            rms = numpy.append(rms, numpy.mean(ps[minFreq:maxFreq]))

    threshold = numpy.amax(rms) * percentageThreshold

    return threshold

# | Computes the absolute value of the raw data values then calculates
# | the mean, max, min, and standard deviation of those data values
def getIntensityFeatures(data):
    abs_val_data = numpy.absolute(data)
    average_intensity = numpy.mean(abs_val_data)
    max = numpy.amax(abs_val_data)
    min = numpy.amin(abs_val_data)
    stDev = numpy.std(abs_val_data)
    return average_intensity, stDev, max, min

# | Computes welch looking back and for number of sampleRate in length of data
# | and returns the average of the loudest pitch in each window
def getAveragePitch(data,sampleRate,windowSize):
    sampleWindowSize = windowSize * sampleRate #windowSize in seconds
    loudest_pitch = numpy.zeros(shape=0)

    step = 0
    while step < len(data):
        freqs, ps = getPowerSpectrum(data[step:step + sampleWindowSize],sampleRate,sampleRate)
        loudest_pitch = numpy.append(loudest_pitch, numpy.argmax(ps))

        #increment to next step
        step += sampleWindowSize
    average_pitch = numpy.mean(loudest_pitch)

    return average_pitch

# | Extracts all features and returns array in accordance with Jamison's drawing
# | Parameters:
# |   - filePath: path to file to process
# |   - stepSize: size of step in seconds
# |   - lookBackSize: size of window for looking back on longterm features in seconds
# |   - percentageThreshold: percentage of loudest noise in band above which vocal activity is present
# |   - printStatus: if True will print out as each second is processed, else no print
def getFeatures(filePath,stepSize,lookBackSize,percentageThreshold,printStatus):
    if printStatus :
        print("[ START ] Working on:",filePath)

    # Read in the file, extract data and metadata
    audioData = wavio.read(filePath) # reads in audio file
    numberOfChannels = len(audioData.data[0])
    sampleRate = audioData.rate # usually 44100
    width = audioData.sampwidth # bit depth is equal to width times 8
    length = len(audioData.data) # gets number of sample in audio

    # make the audio data mono
    data = numpy.mean(audioData.data,axis=1)

    #set up arrays for features
    seconds = numpy.zeros(shape=0)
    voiceActivity = numpy.zeros(shape=0)
    wpm = numpy.zeros(shape=0)
    pitch = numpy.zeros(shape=0)
    meanIntensity = numpy.zeros(shape=0)
    stDevIntensity = numpy.zeros(shape=0)
    maxIntensity = numpy.zeros(shape=0)
    minIntensity = numpy.zeros(shape=0) # this will just be all zeroes

    if printStatus :
        print("Determining threshold for speech...")

    threshold = getVADThreshold(data,           #audio data
                                sampleRate=sampleRate,
                                minFreq=300,    #minimum frequency of vocal activity
                                maxFreq=3000,   #max frequency of vocal activity
                                percentageThreshold=percentageThreshold
                                )
    if printStatus :
        print("Done!")

    step = 0
    sampleStepSize = int(stepSize * sampleRate)
    sampleLookBackSize = int(lookBackSize * sampleRate)
    while step < length:
        if printStatus :
            print("[",str(step/length*100)[:4],"% ] Second",int(step/sampleRate))

        # cut out the chunk to look at
        chunk = data[step:step + sampleStepSize]

        # keep track of what second we're in
        seconds = numpy.append(seconds,step/sampleRate)

        # look backward to calculate features over long term
        if step + sampleStepSize - sampleLookBackSize > 0:

            lookBackChunk = data[step + sampleStepSize - sampleLookBackSize:step + sampleStepSize]

            ### VAD


            ### WORDS PER MINUTE
            transcript = getTranscriptFromData(lookBackChunk,sampleRate,filePath)
            current_wpm = len(transcript)/( lookBackSize/60 )
            wpm = numpy.append(wpm,current_wpm)

            ### AVERAGE PITCH
            pitch = numpy.append(pitch, getAveragePitch(lookBackChunk,sampleRate,1))

            ## INTENSITY FEATURES
            average_intensity, stDev, max, min = getIntensityFeatures(lookBackChunk)
            meanIntensity = numpy.append(meanIntensity, average_intensity)
            stDevIntensity = numpy.append(stDevIntensity, stDev)
            maxIntensity = numpy.append(maxIntensity, max)
            minIntensity = numpy.append(minIntensity, min)

        # fills arrays with zeros until step is larger than lookBackSize
        else:

            wpm = numpy.append(wpm,0)
            pitch = numpy.append(pitch,0)
            meanIntensity = numpy.append(meanIntensity, 0)
            stDevIntensity = numpy.append(stDevIntensity, 0)
            maxIntensity = numpy.append(maxIntensity, 0)
            minIntensity = numpy.append(minIntensity, 0)

        # increment to next step
        step += sampleStepSize

    features = numpy.vstack([seconds,voiceActivity,wpm,pitch,meanIntensity,stDevIntensity,maxIntensity,minIntensity])

    if printStatus :
        print("[ DONE ] Finished processing",filePath,"!")

    return features

def main():
    path1 = "./media/pilot/pilot_1_ol.wav"
    path2 = "./media/Speech16.wav"
    featureArray = getFeatures(filePath=path2, #file
                               stepSize=1, #how big to step
                               lookBackSize=30,  #how big of interval to wait until looking for transcript, pitch/intensity features
                               percentageThreshold=0.001,  #percentage of loudest noise in band above which vocal activity is present
                               printStatus=True
                               )
    print(featureArray)

if __name__ == "__main__":
    main()
