import math
import numpy as np
import tensorflow as tf
from scipy import signal
from hparams import hparams

import librosa
import librosa.filters
import soundfile as sf 

def load_audio(path, pre_silence_length=0, post_silence_length=0):
    audio = librosa.core.load(path, sr=8000)[0]
    if pre_silence_length > 0 or post_silence_length > 0:
        audio = np.concatenate([
                get_silence(pre_silence_length),
                audio,
                get_silence(post_silence_length),
        ])
    return audio

def save_audio(audio, path, sample_rate=None):
    audio *= 32767 / max(0.01, np.max(np.abs(audio)))
    # librosa.output.write_wav(path, audio.astype(np.int16),
            # 8000 if sample_rate is None else sample_rate)

    sf.write(path, audio.astype(np.int16), 8000)

    print(" [*] Audio saved: {}".format(path))


def resample_audio(audio, target_sample_rate):
    return librosa.core.resample(
            audio, 8000, target_sample_rate)


def get_duration(audio):
    return librosa.core.get_duration(audio, sr=8000)


def frames_to_hours(n_frames):
    return sum((n_frame for n_frame in n_frames)) * \
            hparams.frame_shift_ms / (3600 * 1000)


def get_silence(sec):
    return np.zeros(8000 * sec)


def spectrogram(y):
    D = _stft(_preemphasis(y))
    S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
    return _normalize(S)


def inv_spectrogram(spectrogram):
    S = _db_to_amp(_denormalize(spectrogram) + hparams.ref_level_db)    # Convert back to linear
    return inv_preemphasis(_griffin_lim(S ** hparams.power))                 # Reconstruct phase


def inv_spectrogram_tensorflow(spectrogram):
    S = _db_to_amp_tensorflow(_denormalize_tensorflow(spectrogram) + hparams.ref_level_db)
    return _griffin_lim_tensorflow(tf.pow(S, hparams.power))


def melspectrogram(y):
    D = _stft(_preemphasis(y))
    S = _amp_to_db(_linear_to_mel(np.abs(D)))
    return _normalize(S)


def inv_melspectrogram(melspectrogram):
    S = _mel_to_linear(_db_to_amp(_denormalize(melspectrogram)))     # Convert back to linear
    return inv_preemphasis(_griffin_lim(S ** hparams.power))            # Reconstruct phase


# Based on https://github.com/librosa/librosa/issues/434
def _griffin_lim(S):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)

    y = _istft(S_complex * angles)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y


def _griffin_lim_tensorflow(S):
    with tf.variable_scope('griffinlim'):
        S = tf.expand_dims(S, 0)
        S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
        y = _istft_tensorflow(S_complex)
        for i in range(hparams.griffin_lim_iters):
            est = _stft_tensorflow(y)
            angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
            y = _istft_tensorflow(S_complex * angles)
        return tf.squeeze(y, 0)


def _stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
    _, hop_length, win_length = _stft_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_tensorflow(signals):
    n_fft, hop_length, win_length = _stft_parameters()
    return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)
  
  
def _istft_tensorflow(stfts):
    n_fft, hop_length, win_length = _stft_parameters()
    return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)

def _stft_parameters():
    n_fft = (hparams.num_freq - 1) * 2
    hop_length = int(hparams.frame_shift_ms / 1000 * 8000)
    win_length = int(hparams.frame_length_ms / 1000 * 8000)
    return n_fft, hop_length, win_length


# Conversions:

_mel_basis = None
_inv_mel_basis = None

def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)

def _mel_to_linear(mel_spectrogram):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis())
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _build_mel_basis():
    n_fft = (hparams.num_freq - 1) * 2
    return librosa.filters.mel(8000, n_fft, n_mels=hparams.num_mels)

def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def _db_to_amp_tensorflow(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

def _preemphasis(x):
    return signal.lfilter([1, -hparams.preemphasis], [1], x)

def inv_preemphasis(x):
    return signal.lfilter([1], [1, -hparams.preemphasis], x)

def _normalize(S):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)

def _denormalize(S):
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db

def _denormalize_tensorflow(S):
    return (tf.clip_by_value(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db

from sys import byteorder
from array import array
import pyaudio

class AudioRecorder:

  def __init__(self):
    self.rate = 8000
    self.threshold = 0.03 # silence threshold {Need to experiment with it}
    self.chunk_size = 4096
    self.format = pyaudio.paFloat32
    self._pyaudio = pyaudio.PyAudio()

  def isSilent(self, data):
    #Returns true if below the silence threshold
    return max(data) < self.threshold

  def normalize(self, data):
    #Averages the volume {Had to mess with a lot of examples to get this right}
    times = float(0.5) / max(abs(i) for i in data)
    r = array('f')
    for i in data:
      r.append(i * times)

    return r

  def trim(self, data):
    # Trim the blanks at the start and end
    def _trim(data):
      started = False
      r = array('f')
      for i in data:
        if not started and abs(i) > self.threshold:
          started = True
          r.append(i)

        elif started:
          r.append(i)

      return r
    #first trim the left side
    data = _trim(data)
    data.reverse()

    #then trim the right side
    data = _trim(data)
    data.reverse()

    return data

  def addSilence(self, data):
    #adds silence to the start and end of 0.1 seconds
    r = array('f', [0 for i in range(int(0.1 * self.rate))])
    r.extend(data)
    r.extend([0 for i in range(int(0.1 * self.rate))])

    return r

  """
  Records words from the microphone using pyaudio in paFloat32 format and 
  16000Hz sampling rate.
  Returns data as an array of signed floats.
  
  """
  def record(self):
    stream = self._pyaudio.open(format=self.format, channels=1, rate=self.rate, input=True, output=True, frames_per_buffer=self.chunk_size)
    numSilent = 0
    started = False
    r = array('f')
   
    while 1:
      #has to be little endian and signed short
      data = array('f', stream.read(self.chunk_size))
      
      if byteorder == 'big':
        data.byteswap()
      
      r.extend(data)
      aud = np.array(r)
    #   print(aud.shape)
    #   if aud.shape == 8192:
    #     print(aud.shape)
    #     yield aud
      silent = self.isSilent(data) #check silent to add blanks
      
      if silent and started:
        numSilent += 1
      
      elif not silent and not started:
        started = True
      
      if started and numSilent > 30: #if there are 30 silences, break. doesnt handle long sentences.
        break
    
    width = self._pyaudio.get_sample_size(self.format)
    stream.stop_stream()
    stream.close()
    r = self.normalize(r) #normalizes the audio
    r = self.trim(r) 
    r = self.addSilence(r)
    
    return r, width

  def terminate(self):
    self._pyaudio.terminate()