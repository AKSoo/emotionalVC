import argparse, os, glob, json
import numpy as np
import torch

from analyzer import pw2wav
import soundfile

import data_utils as util
from vawgan import Encoder, Generator, forward_VAE


def main(architecture, corpus, tgt, load_dir='save',
         src_path=None, out_dir='output', device=0, **kwargs):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(architecture) as f:
        arch = json.load(f)

    emotions = util.emotion_list()

    # input params
    if src_path is None:
        src_path = arch['training']['src_dir']
    src_files = sorted(glob.glob(src_path))
    normalizer = util.Tanhize(corpus)
    batch_size = arch['training']['batch_size']

    # load models
    length = arch['hwc'][0]
    z_dim, y_dim = arch['z_dim'], arch['y_dim']
    archE = arch['encoder']
    archG = arch['generator']

    E = Encoder(length, z_dim,
                archE['output'], archE['kernel'], archE['stride'])
    G = Generator(length, z_dim, y_dim, archG['merge_dim'], archG['init_out'],
                  archG['output'], archG['kernel'], archG['stride'])
    E.load_state_dict(torch.load(os.path.join(load_dir, 'VAWGAN_encoder.pth')))
    G.load_state_dict(torch.load(os.path.join(load_dir, 'VAWGAN_generator.pth')))

    E.eval()
    G.eval()

    # GPU?
    if device >= 0:
        E.cuda(device)
        G.cuda(device)

    # VAE on each file
    for f in src_files:
        print('Processing {}'.format(f))
        loader = util.load_single(f, normalizer, batch_size=batch_size)
        outputs = []

        for load in loader:
            s_x, s_cond = util.load_to_vars(load, device)
            t_cond = torch.full_like(s_cond, emotions.index(tgt))

            t_xh, _ = forward_VAE(E, G, s_x, t_cond)
            outputs.append(t_xh.cpu().data.numpy())

        src = emotions[s_cond[0].item()]
        sp = np.concatenate(outputs)

        # compile to wav
        _, ap, f0, en, _ = util.get_features(f)
        f0 = util.convert_f0(f0, src, tgt)

        feat = np.concatenate([sp, ap, f0[:, np.newaxis], en[:, np.newaxis]], axis=1)
        y = pw2wav(feat)

        # save
        name = os.path.splitext(os.path.basename(f))[0]
        out_file = os.path.join(out_dir, '{}-{}-{}.wav'.format(src, tgt, name))
        soundfile.write(out_file, y, 22050)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('architecture', type=str, help='json architecture file')
    parser.add_argument('corpus', type=str, help='dataset name')
    parser.add_argument('tgt', type=str, help='target emotion')
    parser.add_argument('--load_dir', type=str, help='trained model directory')
    parser.add_argument('--src_path', type=str, help='source path (default from json)')
    parser.add_argument('--out_dir', type=str, help='output directory')
    parser.add_argument('--device', type=int, help='-1: cpu, 0+: gpu')

    args = {k:v for k,v in vars(parser.parse_args()).items() if v is not None}
    main(**args)
