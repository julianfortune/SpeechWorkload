import wave
from deepspeech import Model
import numpy
import os
import sys

files = []
for f in os.listdir():
    if(os.path.isfile(f) and str(f)[-3:]=='wav'):
        files.append(f)

model_dir = '/scratch/beta_v0.5'
ds_grp = os.path.join(model_dir,'output_graph.pbmm')
ds_alp = os.path.join(model_dir,'alphabet.txt')
ds_lm = os.path.join(model_dir,'lm.binary')
ds_trie = os.path.join(model_dir,'trie')
BEAM_WIDTH = 500
LM_WEIGHT = 1.75
WORD_COUNT_WEIGHT = 1.00
VALID_WORD_COUNT_WEIGHT = 1.00
N_FEATURES = 26
N_CONTEXT = 9

ds = Model(ds_grp, N_FEATURES, N_CONTEXT, ds_alp, BEAM_WIDTH)
ds.enableDecoderWithLM(ds_alp, ds_lm, ds_trie, LM_WEIGHT, WORD_COUNT_WEIGHT)

correct = 0
for f in files:
    audio = None
    with wave.open(f,'rb') as wav:
        fsr = wav.getframerate()
        if(fsr != 16000):
            print("WARNING: Sample rate is not 16000.",file=sys.stderr)
            fsr, audio = convert_samplerate(f)
        else:
            audio = numpy.frombuffer(wav.readframes(wav.getnframes()),numpy.int16)
    output = ds.stt(audio,fsr)

    if('um' in output or 'uh' in output):
        correct+=1
    print(output)

print('Accuracy: {} ({}/{})'.format(correct/len(files),correct,len(files)))

