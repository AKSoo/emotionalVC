import argparse, os
import json
import torch
from torch.autograd import Variable

import data_utils as util
import vawgan


def main(architecture, corpus='savee', device=0, **kwargs):
    with open(architecture) as f:
        arch = json.load(f)

    # load data
    src_path = arch['training']['src_dir']
    tgt_path = arch['training']['trg_dir']
    batch_size = arch['training']['batch_size']

    normalizer = util.Tanhize(corpus)
    src_loader = util.load_specenv(src_path, transform=normalizer, num_workers=4,
                                   batch_size=batch_size, shuffle=True)
    tgt_loader = util.load_specenv(tgt_path, transform=normalizer, num_workers=4,
                                   batch_size=batch_size, shuffle=True)

    # load models
    length = arch['hwc'][0]
    z_dim, y_dim = arch['z_dim'], arch['y_dim']
    archE = arch['encoder']
    archG = arch['generator']
    archD = arch['discriminator']

    E = vawgan.Encoder(length, z_dim, archE['output'], archE['kernel'], archE['stride'])
    G = vawgan.Generator(z_dim, y_dim, archG['merge_dim'],
                         archG['output'], archG['kernel'], archG['stride'])
    D = vawgan.Discriminator(length, y_dim, archD['output'], archD['kernel'], archD['stride'])

    # optimizers
    lr = arch['training']['lr']

    optim_E = torch.optim.Adam(E.parameters(), lr, weight_decay=archE['l2-reg'])
    optim_G = torch.optim.Adam(G.parameters(), lr, weight_decay=archG['l2-reg'])
    optim_D = torch.optim.Adam(D.parameters(), lr, weight_decay=archD['l2-reg'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('architecture', type=str, help='json architecture file')
    parser.add_argument('--corpus', type=str, help='dataset name')
    parser.add_argument('--device', type=int, help='-1: cpu, 0+: gpu')

    args = {k:v for k,v in vars(parser.parse_args()).items() if v is not None}
    main(**args)
