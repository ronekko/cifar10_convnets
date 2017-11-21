# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 16:06:25 2017

@author: sakurai

Implementation of "Wide Residual Networks".
"""

from types import SimpleNamespace

import chainer
import chainer.functions as F
import chainer.links as L

import common
from functions import extend_channels
from links import BRCChain


class WideResnet(chainer.Chain):
    '''
    Args:
        n (int):
            Number of blocks in each group.

        k (int):
            Widening factor.
    '''
    def __init__(self, n=4, k=10):
        super(WideResnet, self).__init__(
            conv1=L.Convolution2D(3, 16, 3, pad=1),
            group2=ResnetGroup(n, 16 * k),
            group3=ResnetGroup(n, 32 * k),
            group4=ResnetGroup(n, 64 * k),
            brc_out=BRCChain(64 * k, 10, 1, pad=0)
        )

    def __call__(self, x):
        h = self.conv1(x)
        h = self.group2(h)
        h = F.max_pooling_2d(h, 2)
        h = self.group3(h)
        h = F.max_pooling_2d(h, 2)
        h = self.group4(h)
        h = self.brc_out(h)
        y = F.average_pooling_2d(h, h.shape[2:]).reshape(-1, 10)
        return y


class ResnetGroup(chainer.ChainList):
    '''Sequence of `ResnetBlock`s.
    '''
    def __init__(self, n_blocks, channels):
        blocks = [ResnetBlock(channels) for i in range(n_blocks)]
        super(ResnetGroup, self).__init__(*blocks)
        self._channels = channels

    def __call__(self, x):
        x = extend_channels(x, self._channels)
        for link in self:
            x = link(x)
        return x


class ResnetBlock(chainer.Chain):
    '''Residual block (y = x + f(x)) of 'pre-activation'.
    '''
    def __init__(self, channels):
        super(ResnetBlock, self).__init__(
            brc1=BRCChain(channels, channels, 3, pad=1),
            brc2=BRCChain(channels, channels, 3, pad=1))

    def __call__(self, x):
        h = self.brc1(x)
        h = self.brc2(h)
        return x + h


if __name__ == '__main__':
    # Hyperparameters
    hparams = SimpleNamespace()
    hparams.gpu = 0  # GPU>=0, CPU < 0
    hparams.n = 4   # number of blocks in each group
    hparams.k = 10  # widening factor
    hparams.num_epochs = 200
    hparams.batch_size = 100
    hparams.optimizer = chainer.optimizers.NesterovAG
    hparams.lr_init = 0.1
    hparams.lr_decrease_rate = 0.2
    hparams.weight_decay = 5e-4
    hparams.epochs_decrease_lr = [60, 120, 160]

    # Model and optimizer
    model = WideResnet(hparams.n, hparams.k)

    result = common.train_eval(model, hparams)

    best_model, best_test_loss, best_test_acc, best_epoch = result[:4]
    train_loss_log, train_acc_log, test_loss_log, test_acc_log = result[4:]
