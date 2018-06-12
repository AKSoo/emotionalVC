import numpy as np
import glob

from torch.utils.data import ConcatDataset, Dataset, DataLoader

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
