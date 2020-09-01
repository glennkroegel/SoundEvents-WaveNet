'''
created_by: Glenn Kroegel
date: 23 August 2023
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pandas as pd
import numpy as np
import os
import sys
import glob
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import normalize
from model import num_classes

audio_dir = 'data/audio/downsampled/'

def make_target_vec(size, info, sr):
    ys = torch.zeros(size).long()
    for event in info:
        t1, t2, label = event
        ix1 = np.round(t1*sr).astype(np.int64)
        ix2 = np.round(t2*sr).astype(np.int64)
        label = int(label)
        ys[ix1:ix2] = label
    return ys

class WavDataset(Dataset):
    def __init__(self, files, info):
        self.files = files
        self.info = info
        # self.transform = torchaudio.transforms.AmplitudeToDB()

    def __getitem__(self, i):
        path = self.files[i]
        basename = os.path.basename(path)
        info = self.info[basename]
        x, sr = torchaudio.load_wav(path)
        x = normalize(x)
        y = make_target_vec(x.shape[1], info, sr)
        return x, y

    def __len__(self):
        return len(self.files)

if __name__ == "__main__":
    files = glob.glob(os.path.join(audio_dir, '*.wav'))
    info = pd.read_pickle('info_dict.pkl')
    random.shuffle(files)
    l = len(files)
    p = 0.7
    ix = int(p * l)
    train_files = files[:ix]
    cv_files = files[ix:]
    train_dataset = WavDataset(train_files, info)
    cv_dataset = WavDataset(cv_files, info)
    train_loader = DataLoader(train_dataset, batch_size=1)
    cv_loader = DataLoader(cv_dataset, batch_size=1)
    torch.save(train_loader, 'train_loader.pt')
    torch.save(cv_loader, 'cv_loader.pt')
    