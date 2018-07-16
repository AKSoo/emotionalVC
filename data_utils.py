import numpy as np
import glob
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from torch.autograd import Variable

SP_DIM = 513
FEAT_DIM = 2*SP_DIM + 2 + 1

def get_features(f):
    '''
    f: bin file path

    Returns: sp, ap, f0, en, labels
    '''
    feat = np.fromfile(f, dtype=np.float32).reshape([-1, FEAT_DIM])
    sp = feat[:, :SP_DIM]
    ap = feat[:, SP_DIM : 2*SP_DIM]
    f0 = feat[:, 2*SP_DIM]
    en = feat[:, 2*SP_DIM + 1]
    labels = feat[:, 2*SP_DIM + 2].astype(int)
    return sp, ap, f0, en, labels

class SpecEnvSet(Dataset):
    '''
    frame spectral envelope and label

    f: bin file path
    transform: normalizer function
    '''
    def __init__(self, f, transform=None):
        self.sp, _, _, _, self.labels = get_features(f)
        self.transform = transform

    def __len__(self):
        return len(self.sp)

    def __getitem__(self, idx):
        sp = self.sp[idx]
        if self.transform is not None:
            sp = self.transform(self.sp[idx])
        return sp, self.labels[idx]

def load_single(f, transform=None, **kwargs):
    '''
    f: bin file path
    transform: normalizer function
    & DataLoader params

    Returns: DataLoader
    '''
    s = SpecEnvSet(f, transform=transform)
    return DataLoader(s, **kwargs)

def load_concat(files, transform=None, **kwargs):
    '''
    files: list of bin file paths
    transform: normalizer function
    & DataLoader params

    Returns: DataLoader
    '''
    bigset = ConcatDataset([SpecEnvSet(f, transform=transform) for f in files])
    return DataLoader(bigset, **kwargs)

def load_to_vars(load, device=-1):
    '''
    puts DataLoader loads into Variables

    Returns: x, cond
    '''
    x, cond = Variable(load[0]), Variable(load[1])
    if device >= 0:
        x, cond = x.cuda(device), cond.cuda(device)
    return x, cond


## ./etc

def emotion_list():
    '''
    Returns: list (lines in etc/speakers.tsv)
    '''
    with open('./etc/speakers.tsv') as f:
        emotions = [line.strip() for line in f]
    return emotions

def convert_f0(f0, src, tgt):
    '''
    Log-normal distribution f0 conversion
    '''
    s_mu, s_std = np.fromfile('./etc/{}.npf'.format(src), np.float32)
    t_mu, t_std = np.fromfile('./etc/{}.npf'.format(tgt), np.float32)

    mask = f0 > 1
    logf0 = np.log(f0[mask])
    logf0 = (logf0 - s_mu) / s_std * t_std + t_mu
    f0[mask] = np.exp(logf0)
    return f0

class Tanhize(object):
    '''
    Normalizes to (-1, 1)

    corpus: string (name in etc/)
    '''
    def __init__(self, corpus):
        self.xmax = np.fromfile('./etc/{}_xmax.npf'.format(corpus)).astype(np.float32)
        self.xmin = np.fromfile('./etc/{}_xmin.npf'.format(corpus)).astype(np.float32)

    def __call__(self, x):
        x = np.clip((x - self.xmin) / (self.xmax - self.xmin), 0, 1)
        return 2*x - 1
