import librosa
import numpy as np
y, sr = librosa.load('/home/le/Downloads/nlp_workspace/NLP_Workspace/Audio/Gender_Classification/Audio_file/.wav',sr = 8000)
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),sr = 8000)
times = librosa.times_like(f0)
print(f0, voiced_flag, voiced_probs)
import matplotlib.pyplot as plt
import librosa.display
fig, ax = plt.subplots()
S_db = librosa.feature.melspectrogram(y)

S_dB = librosa.power_to_db(S_db, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=8000,
                        ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
# ax.set(title='Mel-frequency spectrogram')
# 
# D = librosa.amplitude_to_db(np.abs(librosa.feature.melspectrogram(y)), ref=np.max)
# fig, ax = plt.subplots()
# img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
ax.set(title='pYIN fundamental frequency estimation')
# fig.colorbar(img, ax=ax, format="%+2.f dB")
ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
ax.legend(loc='upper right')
plt.show()