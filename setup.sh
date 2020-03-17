# Utilities
python3 -m pip install numpy==1.16.3
python3 -m pip install matplotlib
python3 -m pip install six

# Features
python3 -m pip install scipy
python3 -m pip install librosa
python3 -m pip install praat-parselmouth # PRAAT wrapper

# Audio manipulation
python3 -m pip install wave
python3 -m pip install wavio
python3 -m pip install pydub

# Microphone recording
python3 -m pip install pyaudio

# Learning
python3 -m pip install pandas==0.25.1
python3 -m pip install tensorflow==1.13.1
python3 -m pip install tflearn==0.3.2

# Issues
# If you see `ImportError: cannot import name 'quote'` make sure `parselmouth`
# is not installed using `python3 -m pip uninstall parselmouth`.
# https://github.com/YannickJadoul/Parselmouth/issues/10
