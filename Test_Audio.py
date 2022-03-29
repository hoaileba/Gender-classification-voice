from Model.ResNet import ResNet
import librosa
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy, scipy, matplotlib.pyplot as plt, IPython.display as ipd, sklearn
import tensorflow as tf
import random
import IPython.display as ipd
from random import randint
import glob
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import pickle
import timeit



#Your statements here


# from DataProcessing.GetSignal import get_segment

def split_audio(audio_data, w, h, threshold_level, tolerence=10):
    split_map = []
    start = 0
    data = np.abs(audio_data)
    threshold = threshold_level*np.mean(data[:25000])
    inside_sound = False
    near = 0
    for i in range(0,len(data)-w, h):
        win_mean = np.mean(data[i:i+w])
        if(win_mean>threshold and not(inside_sound)):
            inside_sound = True
            start = i
        if(win_mean<=threshold and inside_sound and near>tolerence):
            inside_sound = False
            near = 0
            split_map.append([start, i])
        if(inside_sound and win_mean<=threshold):
            near += 1
    return split_map
DIR_ = "/data3/smartcall/hocnv/covid/aicv115m_public_train/train_audio_files_8k/train_audio_files_8k/"
def get_segment(signal ,TARGET_LEN = 8000, sr = 8000):
    batch_X_train = []
#     for wav_file in (list_file): 
#     signal, _ = librosa.load(DIR_+wav_file, sr = sr)
    len_signal = signal.shape[0]
    list_split = librosa.effects.split(signal,top_db=5, ref=np.max)
    len_split_cough = len(list_split)
    rand = randint(0, len_split_cough-1)
    start = list_split[rand][0]
    end = list_split[rand][1]

    #padding
    len_signal = signal.shape[0]
    len_cough = end - start
    len_padding = TARGET_LEN - len_cough
    end_remain = len_signal - end
    start_remain = start

    
#         print (len_signal, start, end, len_padding)
    ## time audio == 1s
    if len_signal == TARGET_LEN:
        signal =  signal
    elif len_padding < 0: # len_cought > 1s
#             print ("A")
        pad_start = randint(start, end-TARGET_LEN)
        pad_end = pad_start + TARGET_LEN
#             print (pad_start, pad_end) 
        signal = signal[pad_start: pad_end]
    elif len_padding >= start_remain + end_remain:
#             print ("B")
#             print (start_remain + end_remain)
        signal = np.tile(signal, int(TARGET_LEN/len_signal))
        len_padding = TARGET_LEN - len(signal)
        pad_start = randint(0,len_padding)
        pad_end = len_padding - pad_start
        signal = np.concatenate((np.zeros(pad_start), signal))
        signal = np.concatenate((signal, np.zeros(pad_end)))
    else:
        if start_remain >= len_padding:
            ind_start = start - len_padding
        else:
            ind_start = 0
        if end_remain <= len_padding:
            pad_start = randint(ind_start, len_signal - TARGET_LEN)
        else:
            pad_start = randint(ind_start, start)
        pad_end = pad_start + TARGET_LEN
#             print (pad_start, pad_end)
        signal = signal[pad_start: pad_end]
    # MFCC
#         mfcc = get_mfcc(signal)
    batch_X_train.append(signal)
    batch_X_train = np.array(batch_X_train)
    return batch_X_train

model = ResNet(input_audio = (64,63,1))
model.load_weights("Weight/Hoai_ResNet_v6.h5")

p ="/home/le/Downloads/nlp_workspace/NLP_Workspace/Audio/Gender_Classification/Denoise/0.wav"

signal = librosa.load(p,sr = 8000)[0]
start = timeit.default_timer()
signal = signal[0:8000]
# print(signal[0].shape)

mfcc = librosa.feature.mfcc(signal, sr = 8000, n_mfcc = 64 ,  hop_length = 128)
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),sr = 8000)
# print(mfcc.shape)
f0 = f0[numpy.logical_not(numpy.isnan(f0))]
print(f0)
mfcc = mfcc.reshape(1,64,63,1)

pred = model.predict(mfcc)
print(pred)
pred = (np.argmax(pred,axis = 1))
if pred[0] == 0:
    print("nam")
else:
    print("nu")
# class Gender_Classification():
#     def __init__(self, Weight_path, shape):
#         self.Weight_path = Weight_path
#         self.shape = shape

#     def get_segment(self,signal ,TARGET_LEN = 8000, sr = 8000):
#         batch_X_train = []
#     #     for wav_file in (list_file): 
#     #     signal, _ = librosa.load(DIR_+wav_file, sr = sr)
#         len_signal = signal.shape[0]
#         list_split = librosa.effects.split(signal,top_db=5, ref=np.max)
#         len_split_cough = len(list_split)
#         rand = randint(0, len_split_cough-1)
#         start = list_split[rand][0]
#         end = list_split[rand][1]

#         #padding
#         len_signal = signal.shape[0]
#         len_cough = end - start
#         len_padding = TARGET_LEN - len_cough
#         end_remain = len_signal - end
#         start_remain = start
#     #         print (len_signal, start, end, len_padding)
#         ## time audio == 1s
#         if len_signal == TARGET_LEN:
#             signal =  signal
#         elif len_padding < 0: # len_cought > 1s
#     #             print ("A")
#             pad_start = randint(start, end-TARGET_LEN)
#             pad_end = pad_start + TARGET_LEN
#     #             print (pad_start, pad_end) 
#             signal = signal[pad_start: pad_end]
#         elif len_padding >= start_remain + end_remain:
#     #             print ("B")
#     #             print (start_remain + end_remain)
#             signal = np.tile(signal, int(TARGET_LEN/len_signal))
#             len_padding = TARGET_LEN - len(signal)
#             pad_start = randint(0,len_padding)
#             pad_end = len_padding - pad_start
#             signal = np.concatenate((np.zeros(pad_start), signal))
#             signal = np.concatenate((signal, np.zeros(pad_end)))
#         else:
#             if start_remain >= len_padding:
#                 ind_start = start - len_padding
#             else:
#                 ind_start = 0
#             if end_remain <= len_padding:
#                 pad_start = randint(ind_start, len_signal - TARGET_LEN)
#             else:
#                 pad_start = randint(ind_start, start)
#             pad_end = pad_start + TARGET_LEN
#     #             print (pad_start, pad_end)
#             signal = signal[pad_start: pad_end]
#         # MFCC
#     #         mfcc = get_mfcc(signal)
#         batch_X_train.append(signal)
#         batch_X_train = np.array(batch_X_train)
#         return batch_X_train

#     def predict(self,model,signal):




stop = timeit.default_timer()

print('Time: ', stop - start)  