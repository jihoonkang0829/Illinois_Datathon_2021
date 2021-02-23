# this is a test file
import librosa
import numpy as numpy
import os
import glob
import pandas as pd

def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

# y, sr = librosa.load(librosa.ex('trumpet'))

recordings_dir = os.getcwd() + '/data/archive/recordings/recordings/'
file_dir = 'afrikaans1.mp3'

y, sr = librosa.load(recordings_dir + file_dir)
arr = librosa.feature.mfcc(y=y, sr=sr)
print(arr.shape)

pd.DataFrame(arr).to_csv('../mfcc_example.csv', index = False)