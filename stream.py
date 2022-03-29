from sys import byteorder
from array import array
import pyaudio
import wave
from Model.ResNet import ResNet
import librosa
import numpy as np
import librosa.display
import random
import glob
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append('..')
import tensorflow as tf
from Denoise.data_tools import scaled_in, inv_scaled_ou
from Denoise.data_tools import audio_files_to_numpy, numpy_audio_to_matrix_spectrogram, matrix_spectrogram_to_numpy_audio
from Denoise.ModelDenoise import unet
import timeit
import soundfile as sf



class Denoise_audio:
    def __init__(self):
        self.sample_rate = 8000
        self.min_duration = 1.0
        self.frame_length = 8064
        self.hop_length_frame = 8064
        self.hop_length_frame_noise = 5000
        self.loaded_model = unet()
        self.path_weights = '/home/le/Downloads/nlp_workspace/NLP_Workspace/Audio/Gender_Classification/Denoise/weights/'
        self.loaded_model.load_weights(self.path_weights+'model_unet.h5')
        self.n_fft= 255
        self.hop_length_fft = 63

    def denoise_audio(self,audio_numpy):
        audio = audio_files_to_numpy(audio_numpy, self.sample_rate,
                             self.frame_length, self.hop_length_frame, self.min_duration)

    
        

        dim_square_spec = int(self.n_fft / 2) + 1

        # Create Amplitude and phase of the sounds
        m_amp_db_audio,  m_pha_audio = numpy_audio_to_matrix_spectrogram(
            audio, dim_square_spec, self.n_fft, self.hop_length_fft)

        #global scaling to have distribution -1/1
        X_in = scaled_in(m_amp_db_audio)
        print(X_in.shape)
        #Reshape for prediction
        X_in = X_in.reshape(X_in.shape[0],X_in.shape[1],X_in.shape[2],1)
        #Prediction using loaded network
        X_pred = self.loaded_model.predict(X_in)
        #Rescale back the noise model
        inv_sca_X_pred = inv_scaled_ou(X_pred)
        #Remove noise model from noisy speech
        X_denoise = m_amp_db_audio - inv_sca_X_pred[:,:,:,0]
        #Reconstruct audio from denoised spectrogram and phase
        audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, self.frame_length, self.hop_length_fft)
        #Number of frames
        nb_samples = audio_denoise_recons.shape[0]


        #Save all frames in one file
        denoise_long = audio_denoise_recons.reshape(1, nb_samples * self.frame_length)*10
        return denoise_long 


class Gender_Classification:

  def __init__(self):
    self.rate = 8000
    self.threshold =    0.03 # silence threshold {Need to experiment with it}
    self.chunk_size = 4032
    self.format = pyaudio.paFloat32
    self._pyaudio = pyaudio.PyAudio()
    self.denoise = Denoise_audio()
    self.model_cls = ResNet(input_audio = (64,64,1))
    self.PATH_WEIGHT_CLS = "/home/le/Downloads/nlp_workspace/NLP_Workspace/Audio/Gender_Classification/Weight/Hoai_ResNet_RemoveNoise_Train1+2.h5"
    self.model_cls.load_weights(self.PATH_WEIGHT_CLS)


  def run_stream(self):
    stream = self._pyaudio.open(format=self.format, channels=1, rate=self.rate, input=True, output=True, frames_per_buffer=self.chunk_size)
    numSilent = 0
    started = False
    sinal_audio = array('f')
    i = 0
    while 1:      
      data = array('f', stream.read(self.chunk_size))
      if byteorder == 'big':
        data.byteswap()
      
      sinal_audio.extend(data)
      audio_numpy = np.array(sinal_audio)
      if audio_numpy.shape[0] == 8064:
        start =timeit.default_timer()
        X_pred = self.denoise.denoise_audio(audio_numpy)
        mfcc = librosa.feature.mfcc(X_pred[0], sr = 8000, n_mfcc = 64,  hop_length = 128)
        mfcc = mfcc.reshape(1,64,64,1)
        pred = self.model_cls.predict(mfcc)
        end =timeit.default_timer()
        print(end-start)
        pred1 = (np.argmax(pred,axis = 1))
        print(pred)

        if pred[0][pred1[0]] < 0.85:
            print("SILENCE")

        elif pred1[0] == 0:
            print("NAM", pred)
            # print((f0))
        else:
            print("NU", pred)
            # print((f0))
        print("-----------END-------------")
        sinal_audio = array('f')

  def run_save_to_file(self):
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
        id = 0
        while True :
            
            signal = []
            frames = []
            while len(signal) < 16000:
                buffer = stream.read(1000)
                tmp = np.frombuffer(buffer, dtype = 'int16')
                # tmp = tmp +0.02

                frames.append(buffer)
                signal = np.concatenate((signal, tmp), axis = -1)

            wf = wave.open(WAVE_OUTPUT_FILENAME+ str(id) + '.wav', 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            pa ="/home/le/Downloads/nlp_workspace/NLP_Workspace/Audio/Gender_Classification/Audio_file/" + str(id) + '.wav'
            signal = librosa.load(pa,sr = 8000)[0]
            start =timeit.default_timer()
            X_pred = self.denoise.denoise_audio(signal)
            mfcc = librosa.feature.mfcc(X_pred[0][0:8000], sr = 8000, n_mfcc = 64,  hop_length = 128)
            mfcc = mfcc.reshape(1,64,63,1)

            pred = self.model_cls.predict(mfcc)
            end =timeit.default_timer()
            print(end-start)
            pred1 = (np.argmax(pred,axis = 1))
            print(pred)
            if pred[0][pred1[0]] < 0.8:
                print("SILENCE")

            elif pred1[0] == 0:
                print("NAM", pred)
            else:
                print("NU", pred)
            print(X_pred.shape)

  def terminate(self):
    self._pyaudio.terminate()


if __name__ == "__main__":
    gender_cls = Gender_Classification()
    gender_cls.run_stream()
