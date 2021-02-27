import librosa
import pandas as pd
import os
import glob
from tqdm import tqdm

CSV_SAVE_DIR = os.getcwd() + '/data/archive/csv/'
PROCESS_LANGUAGE_LIST = ['vietnamese', 'english', 'turkish', 'arabic', 'korean', 'cantonese', 'farsi', 'japanese', 'mandarin']
print(CSV_SAVE_DIR)

recordings_dir = os.getcwd() + '/data/archive/recordings/recordings/'
mp3_list = glob.glob(recordings_dir + '*.mp3')

for file_dir in tqdm(mp3_list):
    if any(lang in file_dir for lang in PROCESS_LANGUAGE_LIST):
        y, sr = librosa.load(file_dir)
        arr = librosa.feature.mfcc(y=y, sr=sr)
        file_name = file_dir.rsplit('/', 1)[-1].split('.mp3', 1)[0] + '.csv'
        pd.DataFrame(arr).to_csv(CSV_SAVE_DIR + file_name, index = False)
