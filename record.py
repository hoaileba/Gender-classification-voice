import pyaudio
import wave
# from Model.ResNet import ResNet
import librosa
import numpy as np
import pandas as pd
# import librosa
# import matplotlib.pyplot as plt
import librosa.display
import numpy, scipy,IPython.display as ipd
import tensorflow as tf
import random
import IPython.display as ipd
from random import randint
import glob
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import pickle
import timeit

import noisereduce as nr


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 8000
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "/home/le/Downloads/nlp_workspace/NLP_Workspace/Audio/Gender_Classification/Audio_file/"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=1024)
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print(len(frames))
wf = wave.open(WAVE_OUTPUT_FILENAME+ str(0) + '.wav', 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()