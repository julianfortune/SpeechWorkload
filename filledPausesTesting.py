import parselmouth
import numpy as np
import matplotlib.pyplot as plt

import wavio
import math
import librosa

windowSize = .1

filePath = "../media/Participant_Audio_30_Sec_Chunks/p13_ol_chunk10.wav"

inputSound = parselmouth.Sound(filePath)

audio = wavio.read(filePath) # reads in audio file
sampleRate = audio.rate
data = np.mean(audio.data,axis=1)

power = np.absolute(data)

formantData = inputSound.to_formant_burg(window_length=windowSize, max_formants=2)
pitch = inputSound.to_pitch_ac()

pitch_values = pitch.selected_array['frequency']
pitch_values[pitch_values==0] = np.nan

pitchTimes = pitch.t_bins()[:, :-1]

times = formantData.t_bins()[:, :-1][:,0]

firstFormant = []
secondFormant = []
intensity = []

for timeStamp in times:
    firstFormant.append(formantData.get_value_at_time(1, timeStamp))
    secondFormant.append(formantData.get_value_at_time(2, timeStamp))

    frame = int(sampleRate * timeStamp)
    intensity.append(power[frame])

firstFormant = np.array(firstFormant)
secondFormant = np.array(secondFormant)
intensity = np.array(intensity)

sampleWindowSize = int(windowSize*sampleRate)
sampleStepSize = int(windowSize*sampleRate/4)

energy = librosa.feature.rmse(data, frame_length=sampleWindowSize, hop_length=sampleStepSize)[0]

length = math.ceil(data.size/sampleRate)

energyTimes = np.arange(0, length + windowSize/4, windowSize/4)

print(energyTimes)

# plt.plot(times, firstFormant, times, secondFormant, energyTimes, energy, pitchTimes, pitch_values*10, 'ro')
# plt.show()

diffOffset = np.empty(firstFormant.size)
diffOffset.fill(700)

plt.plot(times, firstFormant, times, secondFormant, times, np.abs(firstFormant - secondFormant) - diffOffset)
plt.show()
