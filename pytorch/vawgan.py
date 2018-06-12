'''
VAE-WGAN with convolution and gradient penalty
'''
import torch
import torch.nn as nn
from math import ceil


def _sanity_check(outputs, kernels, strides):
    assert len(outputs) == len(kernels) == len(strides) > 2

def convblock(c_in, c_out, kernel, stride, pad=True, L=0, transpose=False,
              activation=nn.LeakyReLU(inplace=True), batchnorm=True):
    if pad:
        # tensorflow SAME padding
        need = -L % stride + kernel - stride
        pad = ceil(need / 2)
    else:
        pad = 0

    layers = []
    if transpose:
        layers.append(nn.ConvTranspose1d(c_in, c_out, kernel, stride, pad))
    else:
        layers.append(nn.Conv1d(c_in, c_out, kernel, stride, pad))
    layers.append(activation)
    if batchnorm:
        layers.append(nn.BatchNorm1d(c_out))

    return nn.Sequential(*layers)


class Encoder(nn.Module):
    '''
    len_in, z_dim: int
    outputs, kernels, strides: list of int

    IN: tensor (batch, len_in)
    OUT: mean, logvar tensors (batch, z_dim)
    '''
    def __init__(self, len_in, z_dim, outputs, kernels, strides):
        super().__init__()
        _sanity_check(outputs, kernels, strides)

        # convs
        blocks = []
        cur_len, cur_out = len_in, 1
        for o, k, s in zip(outputs, kernels, strides):
            blocks.append(convblock(cur_out, o, k, s, L=cur_len))
            cur_len, cur_out = ceil(cur_len / s), o

        self.blocks = nn.Sequential(*blocks)

        # Gaussian stats
        self.fc_mean = nn.Linear(cur_len * cur_out, z_dim)
        self.fc_logvar = nn.Linear(cur_len * cur_out, z_dim)

    def forward(self, x):
        x = x.unsqueeze(1) # N, 1, L
        x = self.blocks(x) # N, C, L
        x = x.view(x.shape[0], -1) # N, L*C
        return self.fc_mean(x), self.fc_logvar(x)


class Generator(nn.Module):
    '''
    z_dim, y_dim, merge_dim, init_c, init_l: int
    outputs, kernels, strides: list of int

    IN: tensor (batch, z_dim), condition tensor (batch,)
    OUT: tensor (batch, len_out)
    '''
    def __init__(self, z_dim, y_dim, merge_dim, init_l, init_c,
                 outputs, kernels, strides):
        super().__init__()
        _sanity_check(outputs, kernels, strides)

        # embed, merge condition
        self.y = nn.Embedding(y_dim, merge_dim)
        self.fc_z = nn.Linear(z_dim, merge_dim)

        # init reshape
        self.init_l, self.init_c = init_l, init_c
        self.fc_reshape = nn.Sequential(
            nn.Linear(2*merge_dim, init_l * init_c),
            nn.LeakyReLU(inplace=True), nn.BatchNorm1d(init_l * init_c)
        )

        # transpose convs
        blocks = []
        cur_out = init_c
        for o, k, s in zip(outputs[:-1], kernels[:-1], strides[:-1]):
            blocks.append(convblock(cur_out, o, k, s, transpose=True))
            cur_out = o

        # no batchnorm, use tanh before output
        blocks.append(convblock(cur_out, outputs[-1], kernels[-1], strides[-1],
                                transpose=True, activation=nn.Tanh(), batchnorm=False))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, z, condition):
        out = torch.cat([self.fc_z(z), self.y(condition)], 1)
        out = nn.functional.leaky_relu(out, inplace=True)

        out = self.fc_reshape(out) # N, L*C
        out = out.view(-1, self.init_c, self.init_l) # N, C, L

        out = self.blocks(out) # N, 1, L
        return out.squeeze(1)


class Discriminator(nn.Module):
    '''
    len_in, y_dim: int
    outputs, kernels, strides: list of int

    IN: tensor (batch, len_in), condition tensor (batch,)
    OUT: tensor (batch,)
    '''
    def __init__(self, len_in, y_dim, outputs, kernels, strides):
        super().__init__()
        _sanity_check(outputs, kernels, strides)

        # embed condition
        self.y = nn.Embedding(y_dim, len_in)

        # convs
        blocks = []
        cur_len, cur_out = len_in, 2
        batchnorm = False # no batchnorm for input block
        for o, k, s in zip(outputs, kernels, strides):
            blocks.append(convblock(cur_out, o, k, s, L=cur_len, batchnorm=batchnorm))
            cur_len, cur_out = ceil(cur_len / s), o
            print(cur_len, cur_out)
            if not batchnorm:
                batchnorm = True

        self.blocks = nn.Sequential(*blocks)

        # output score
        self.fc_score = nn.Linear(cur_len * cur_out, 1)

    def forward(self, x, condition):
        x = x.unsqueeze(1)
        y = self.y(condition).unsqueeze(1)
        out = torch.cat([x, y], 1) # N, 2, L

        out = self.blocks(out) # N, C, L

        out = out.view(x.shape[0], -1) # N, L*C
        out = self.fc_score(out) # N, 1
        return out.squeeze(1)
