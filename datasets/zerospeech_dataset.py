import json
import numpy as np
import os
import torch
import torch.nn.functional
import soundfile as sf
from torch.utils.data import Dataset
import h5py
import pickle
import logging
logger = logging.getLogger(__name__)


class ZerospeechDataset(Dataset):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--task_input_dir", type=str, default="/work/vjsalt22/dataset/zerospeech2021")
        parser.add_argument("--audio_feat_len", type=float, help="maximal audio length", default=8)
        parser.add_argument("--val_audio_feat_len", type=float, help="maximal audio length", default=10.)

    def __init__(self, args, split, data_source):
        self.args = args
        self.split = split
        self.data_source = data_source
        self.audio_feat_len = args.audio_feat_len if "train" in split else args.val_audio_feat_len
        self.path_txt =  os.path.join(self.args.task_input_dir, f"{split}_{data_source}.txt")
        
        with open(self.path_txt, "r") as f:
            data = [os.path.join(self.args.task_input_dir, split, data_source, x.strip()) for x in f.readlines() if x.strip().endswith(".wav")]
            self.audio_wav_paths = data

    def _LoadAudio(self, path):
        x, sr = sf.read(path, dtype = 'float32')
        assert sr == 16000
        length_orig = len(x)
        #if length_orig > 16000 * self.audio_feat_len:
        #    print('longer')
        #    audio_length = int(16000 * self.audio_feat_len)
        #    x = x[:audio_length]
        #    x_norm = (x - np.mean(x)) / np.std(x)
        #    x = torch.FloatTensor(x_norm) 
        #else:
        audio_length = length_orig
        #    new_x = torch.zeros(int(16000 * self.audio_feat_len))
        x_norm = (x - np.mean(x)) / np.std(x)
        #    new_x[:audio_length] = torch.FloatTensor(x_norm) 
        #    x = new_x
        x = torch.FloatTensor(x_norm)
        return x, audio_length

    def __getitem__(self, index):
        wavpath = self.audio_wav_paths[index]
        audio, nframes = self._LoadAudio(wavpath)
        return audio, nframes, wavpath

    def __len__(self):
        return len(self.audio_wav_paths)

    def collate(self, batch):
        vals = list(zip(*batch))
        collated = {}
        collated['audio'] = torch.nn.utils.rnn.pad_sequence(vals[0], batch_first=True)
        collated['audio_length'] = torch.LongTensor(vals[1])
        collated['audio_attention_mask'] = torch.arange(len(collated['audio'][0])).unsqueeze(0) >= collated['audio_length'].unsqueeze(1)
        collated['path'] = vals[2]
        return collated

