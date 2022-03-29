import librosa

import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
import glob
import re
speechFileList = ["/home/le/Downloads/nlp_workspace/NLP_Workspace/Audio/Gender_Classification/Audio_file/0.wav"]
for speechFile in speechFileList:
  y, sr = librosa.load(speechFile,sr=8000)
#   m = re.match(r'\./C/(.+)\.wav', speechFile)
  filename = "1"
  print(filename)
  #file.write(filename + '\n')
  path = 'segment/{}'.format(filename)
  print(path)
  if not os.path.isdir(path):
    os.makedirs(path)
#   librosa.display.waveplot(y,sr)
  plt.savefig('{}/waveform_origin.png'.format(path))
  plt.clf()
#   m = re.match(r'\./C/(.+)\.wav', speechFile)  
  ts = librosa.effects.split(y,top_db=15, ref=np.max)
  i = 1
  log_file = '{}/log.txt'.format(path)
  with open(log_file,'w') as file:
    for start_i, end_i in ts:
      print('chunk {} in file {}'.format(i, filename))
      file.write('chunk {} in file {}\n'.format(i, filename))
      #print('time: {}s'.format(float(end_i-start_i+1)/sr))
      file.write('time: {}s\n'.format(float(end_i-start_i+1)/8000))
      plt.subplot(len(ts),1,i)
      librosa.display.waveplot(y[start_i:end_i],8000)
    #   librosa.output.write_wav('{}/segment{}.wav'.format(path,i),y[start_i:end_i],sr)
      sf.write('{}/segment{}.wav'.format(path,i), y[start_i:end_i], 8000)
      i = i+1
  plt.autoscale()
  plt.savefig('{}/waveform.png'.format(path))
  plt.clf()