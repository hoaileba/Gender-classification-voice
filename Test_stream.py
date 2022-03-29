import pyaudio
import wave
from Model.ResNet import ResNet
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
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import pickle
import timeit

import noisereduce as nr

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
model.load_weights("Weight/Hoai_ResNet_v7.h5")



CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 8000
RECORD_SECONDS = 1.5
WAVE_OUTPUT_FILENAME = "/home/le/Downloads/nlp_workspace/NLP_Workspace/Audio/Gender_Classification/Audio_file/"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=1000)

print("* recording")
# def record(duration=3, fs=8000):
#     nsamples = duration*fs
#     p = pyaudio.PyAudio()
#     stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True,
#                     frames_per_buffer=nsamples)
#     buffer = stream.read(nsamples)
#     array = np.frombuffer(buffer, dtype='int16')
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#     return array
# a = record()
# print(a.shape)
# nsamples = 12000
# p = pyaudio.PyAudio()
# stream = p.open(format=pyaudio.paInt16, channels=1, rate=8000, input=True,
#                     frames_per_buffer=nsamples)

id = 0
while True :
    # frames = []
    # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    #     data = stream.read(CHUNK)
    #     frames.append(data)
    # print(len(frames))
    # wf = wave.open(WAVE_OUTPUT_FILENAME+ str(id) + '.wav', 'wb')
    # wf.setnchannels(CHANNELS)
    # wf.setsampwidth(p.get_sample_size(FORMAT))
    # wf.setframerate(RATE)
    # wf.writeframes(b''.join(frames))
    # wf.close()
    # pa ="/home/hoaileba/NLP_Workspace/Audio/Gender_Classification/Audio_file/output" + str(id) + '.wav'
    # signal = librosa.load(pa,sr = 8000)[0]
    signal = []
    frames = []
    while len(signal) < 16000:
        buffer = stream.read(1000)
        tmp = np.frombuffer(buffer, dtype = 'int16')
        # tmp = tmp +0.02

        frames.append(buffer)
        signal = np.concatenate((signal, tmp), axis = -1)
    # print(signal)
    
    # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        
    # print(len(frames))
    wf = wave.open(WAVE_OUTPUT_FILENAME+ str(id) + '.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    pa ="/home/le/Downloads/nlp_workspace/NLP_Workspace/Audio/Gender_Classification/Audio_file/" + str(id) + '.wav'
    signal = librosa.load(pa,sr = 8000)[0]
    start = timeit.default_timer()
    X_pred = loaded_model.predict(X_in)
    stop = timeit.default_timer()
    print("time: ", stop - start)

    #Rescale back the noise model
    inv_sca_X_pred = inv_scaled_ou(X_pred)
    #Remove noise model from noisy speech
    X_denoise = m_amp_db_audio - inv_sca_X_pred[:,:,:,0]
    #Reconstruct audio from denoised spectrogram and phase
    audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, frame_length, hop_length_fft)
    #Number of frames
    nb_samples = audio_denoise_recons.shape[0]


    #Save all frames in one file
    denoise_long = audio_denoise_recons.reshape(1, nb_samples * frame_length)*10
    # signal = nr.reduce_noise(audio_clip=signal,noise_clip=signal, verbose=False) 
    # print(signal.shape)
    # break
    start = timeit.default_timer()
    # signal = get_segment(signal, TARGET_LEN = 8000)
    # print(signal.shape)
    id+=1
    mfcc = librosa.feature.mfcc(signal, sr = 8000, n_mfcc = 64,  hop_length = 128)
    # print(mfcc.shape)
    mfcc = mfcc.reshape(1,64,63,1)

    pred = model.predict(mfcc)
    pred1 = (np.argmax(pred,axis = 1))
    print(pred)
    if pred[0][pred1[0]] < 0.65:
        print("SILENCE")

    elif pred1[0] == 0:
        print("NAM", pred)
    else:
        print("NU", pred)
print(len(frames))

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()




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




# stop = timeit.default_timer()

# print('Time: ', stop - start)  