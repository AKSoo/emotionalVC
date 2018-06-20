import argparse, os
import json
import numpy as np
import torch

import data_utils as util
from vawgan import *

# untuned hyperparam
LAMBDA_GP = 10


def main(architecture, corpus, device=0, logdir='logdir', **kwargs):
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    with open(architecture) as f:
        arch = json.load(f)

    # load data
    src_path = arch['training']['src_dir']
    tgt_path = arch['training']['trg_dir']
    batch_size = arch['training']['batch_size']

    normalizer = util.Tanhize(corpus)
    src_loader = util.load_specenv(src_path, transform=normalizer,
                                   batch_size=batch_size, shuffle=True, drop_last=True)
    tgt_loader = util.load_specenv(tgt_path, transform=normalizer,
                                   batch_size=batch_size, shuffle=True, drop_last=True)

    # load models
    length = arch['hwc'][0]
    z_dim, y_dim = arch['z_dim'], arch['y_dim']
    archE = arch['encoder']
    archG = arch['generator']
    archD = arch['discriminator']

    E = Encoder(length, z_dim,
                archE['output'], archE['kernel'], archE['stride'])
    G = Generator(length, z_dim, y_dim, archG['merge_dim'], archG['init_out'],
                  archG['output'], archG['kernel'], archG['stride'])
    D = Discriminator(length, y_dim,
                      archD['output'], archD['kernel'], archD['stride'])

    # GPU?
    if device >= 0:
        E.cuda(device)
        G.cuda(device)
        D.cuda(device)

    # optimizers
    lr = arch['training']['lr']

    optim_E = torch.optim.Adam(E.parameters(), lr, weight_decay=archE['l2-reg'])
    optim_G = torch.optim.Adam(G.parameters(), lr, weight_decay=archG['l2-reg'])
    optim_D = torch.optim.Adam(D.parameters(), lr, weight_decay=archD['l2-reg'])

    ## PRETRAIN VAE ##
    num_iter = min(len(src_loader), len(tgt_loader))
    epoch_vae = arch['training']['epoch_vae']

    for epoch in range(epoch_vae):
        recon_losses = np.zeros(num_iter)
        kld_losses = np.zeros(num_iter)

        for i, load in enumerate(zip(src_loader, tgt_loader)):
            s_x, s_cond, t_x, t_cond = util.load_variables(load, device)

            # forward
            s_xh, (s_mu, s_logv) = forward_VAE(E, G, s_x, s_cond)
            t_xh, (t_mu, t_logv) = forward_VAE(E, G, t_x, t_cond)

            recon = (recon_loss(s_x, s_xh) + recon_loss(t_x, t_xh)) / 2
            kld = (kld_loss(s_mu, s_logv) + kld_loss(t_mu, t_logv)) / 2

            # backward
            optim_E.zero_grad()
            optim_G.zero_grad()
            (recon + kld).backward()
            optim_E.step()
            optim_G.step()

            # stats
            recon_losses[i] = recon.item()
            kld_losses[i] = kld.item()

        print('VAE: Epoch [{:3d}/{:3d}] recon= {:.3f} kld= {:.3f}'
              .format(epoch+1, epoch_vae, np.mean(recon_losses), np.mean(kld_losses)))

    # save VAE checkpoint
    torch.save(E.state_dict(), os.path.join(logdir, 'VAE_encoder.pth'))
    torch.save(G.state_dict(), os.path.join(logdir, 'VAE_generator.pth'))
    print('Saved VAE to ' + logdir + '\n')

    ## TRAIN VAWGAN ##
    epoch_vawgan = arch['training']['epoch_vawgan']
    n_unroll = arch['training']['n_unroll']
    n_unroll_intense = arch['training']['n_unroll_intense']
    gamma_wgan = arch['training']['gamma']

    for epoch in range(epoch_vawgan):
        recon_losses = np.zeros(num_iter)
        kld_losses = np.zeros(num_iter)
        wgan_losses = np.zeros(num_iter)

        for i, load in enumerate(zip(src_loader, tgt_loader)):
            s_x, s_cond, t_x, t_cond = util.load_variables(load, device)

            # train discriminator first
            s2t_xh, _ = forward_VAE(E, G, s_x, t_cond)

            wgan = wgan_loss(D, t_x, s2t_xh, t_cond)
            grad_penalty = gradient_penalty(D, t_x, s2t_xh, t_cond)

            optim_D.zero_grad()
            (-wgan + LAMBDA_GP * grad_penalty).backward()
            optim_D.step()

            # train generator every n_unroll
            if i > 0 and i % n_unroll == 0 or i == 0 and i % n_unroll_intense == 0:
                s_xh, (s_mu, s_logv) = forward_VAE(E, G, s_x, s_cond)
                t_xh, (t_mu, t_logv) = forward_VAE(E, G, t_x, t_cond)
                s2t_xh, _ = forward_VAE(E, G, s_x, t_cond)

                recon = (recon_loss(s_x, s_xh) + recon_loss(t_x, t_xh)) / 2
                kld = (kld_loss(s_mu, s_logv) + kld_loss(t_mu, t_logv)) / 2
                wgan = wgan_loss(D, t_x, s2t_xh, t_cond)

                optim_E.zero_grad()
                (recon + kld).backward(retain_graph=True)
                optim_E.step()

                optim_G.zero_grad()
                (recon + gamma_wgan * wgan).backward()
                optim_G.step()

            # stats
            recon_losses[i] = recon.item()
            kld_losses[i] = kld.item()
            wgan_losses[i] = wgan.item()

        print('VAWGAN: Epoch [{:3d}/{:3d}] recon= {:.3f} kld= {:.3f} wgan= {:.3f}'
              .format(epoch+1, epoch_vawgan, np.mean(recon_losses), np.mean(kld_losses),
                      np.mean(wgan_losses)))

    # save VAWGAN
    torch.save(E.state_dict(), os.path.join(logdir, 'VAWGAN_encoder.pth'))
    torch.save(G.state_dict(), os.path.join(logdir, 'VAWGAN_generator.pth'))
    torch.save(D.state_dict(), os.path.join(logdir, 'VAWGAN_discriminator.pth'))
    print('Saved VAWGAN to ' + logdir + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('architecture', type=str, help='json architecture file')
    parser.add_argument('corpus', type=str, help='dataset name')
    parser.add_argument('--device', type=int, help='-1: cpu, 0+: gpu')
    parser.add_argument('--logdir', type=str, help='save directory')

    args = {k:v for k,v in vars(parser.parse_args()).items() if v is not None}
    main(**args)
