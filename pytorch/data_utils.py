import numpy as np
import glob
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from torch.autograd import Variable

SP_DIM = 513
FEAT_DIM = 2*SP_DIM + 2 + 1

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

class SpecEnvSet(Dataset):
    '''
    frame spectral envelope and label

    bin_path: bin file
    transform: normalizer function
    '''
    def __init__(self, bin_path, transform=None):
        feat = np.fromfile(bin_path, dtype=np.float32).reshape([-1, FEAT_DIM])
        self.sp = feat[:, :SP_DIM]
        self.labels = feat[:, -1].astype(int)

        self.transform = transform

    def __len__(self):
        return len(self.sp)

    def __getitem__(self, idx):
        sp = self.sp[idx]
        if self.transform is not None:
            sp = self.transform(self.sp[idx])

        return sp, self.labels[idx]

def load_specenv(path, transform=None, **kwargs):
    '''
    path: bin files pattern
    transform: normalizer function
    & DataLoader params

    Returns: DataLoader
    '''
    files = sorted(glob.glob(path))
    spset = ConcatDataset([SpecEnvSet(f, transform=transform) for f in files])
    return DataLoader(spset, **kwargs)

def load_variables(load, device=0):
    '''
    loads from DataLoaders into Variables

    Returns: s_x, s_cond, t_x, t_cond
    '''
    s, t = load
    s_x, s_cond = Variable(s[0]), Variable(s[1])
    t_x, t_cond = Variable(t[0]), Variable(t[1])

    if device >= 0:
        s_x, s_cond = s_x.cuda(device), s_cond.cuda(device)
        t_x, t_cond = t_x.cuda(device), t_cond.cuda(device)

    return s_x, s_cond, t_x, t_cond
