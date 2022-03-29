import os
import re
import sys
import json
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob


from hparams import hparams
from audio import load_audio, save_audio, get_duration, get_silence
# from utils import add_postfix
def add_postfix(path, postfix):
    path_without_ext, ext = path.rsplit('.', 1)
    return "{}.{}.{}".format(path_without_ext, postfix, ext)
def abs_mean(x):
    return abs(x).mean()

def remove_breath(audio):
    edges = librosa.effects.split(
            audio, top_db=40, frame_length=128, hop_length=32)

    for idx in range(len(edges)):
        start_idx, end_idx = edges[idx][0], edges[idx][1]
        if start_idx < len(audio):
            if abs_mean(audio[start_idx:end_idx]) < abs_mean(audio) - 0.05:
                audio[start_idx:end_idx] = 0

    return audio

def split_on_silence_with_librosa(
        audio_path, top_db=40, frame_length=1024, hop_length=256,
        skip_idx=0, out_ext="wav",
        min_segment_length=3, max_segment_length=8,
        pre_silence_length=0, post_silence_length=0):

    filename = os.path.basename(audio_path).split('.', 1)[0]
    in_ext = audio_path.rsplit(".")[1]

    audio = load_audio(audio_path)

    edges = librosa.effects.split(audio,
            top_db=top_db, frame_length=frame_length, hop_length=hop_length)

    new_audio = np.zeros_like(audio)
    for idx, (start, end) in enumerate(edges[skip_idx:]):
        new_audio[start:end] = remove_breath(audio[start:end])
        
    save_audio(new_audio, add_postfix(audio_path, "no_breath"))
    audio = new_audio
    edges = librosa.effects.split(audio,
            top_db=top_db, frame_length=frame_length, hop_length=hop_length)

    audio_paths = []
    for idx, (start, end) in enumerate(edges[skip_idx:]):
        segment = audio[start:end]
        duration = get_duration(segment)

        if duration <= min_segment_length or duration >= max_segment_length:
            continue

        output_path = "{}/{}.{:04d}.{}".format(
                os.path.dirname(audio_path), filename, idx, out_ext)

        padded_segment = np.concatenate([
                get_silence(pre_silence_length),
                segment,
                get_silence(post_silence_length),
        ])


        
        save_audio(padded_segment, output_path)
        audio_paths.append(output_path)

    return audio_paths



path = split_on_silence_with_librosa(audio_path= '/home/le/Downloads/nlp_workspace/NLP_Workspace/Audio/Gender_Classification/record (2).wav')
print(path)