# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 19:14:24 2017

@author: sakurai

A Chainer implementation of DenseNet,
"Densely Connected Convolutional Networks"
https://arxiv.org/abs/1608.06993v3
"""

from types import SimpleNamespace
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

import common
from links import BRCChain


class DensenetBC(chainer.ChainList):
    '''
    Args:
        nums_units (list of int):
            List of numbers of primitive functions, for each of dense-blocks.

        growth_rate (int):
            Output channels of each primitive H(x), i.e. `k`.
    '''
    def __init__(self, num_classes=10, nums_units=[20, 20, 20], growth_rate=12,
                 dropout_rate=0.2, compression_factor=0.5):
        out_channels = growth_rate * 2
        funcs = [L.Convolution2D(None, out_channels, 3, pad=1, nobias=True)]
        for num_units in nums_units:
            in_channels = out_channels
            funcs.append(DenseBlockBC(in_channels, num_units, growth_rate,
                                      dropout_rate))
            in_channels += growth_rate * num_units
            out_channels = int(np.ceil(in_channels * compression_factor))
            funcs.append(TransitionLayer(in_channels, out_channels))
        funcs.pop(-1)  # in order to replace the last one with global pooling
        funcs.append(
            TransitionLayer(in_channels, num_classes, global_pool=True))
        super(DensenetBC, self).__init__(*funcs)
        self._num_classes = num_classes

    def __call__(self, x):
        conv1 = self[0]
        blocks = self[1:]  # dense, transition, ..., dense, transision

        h = conv1(x)
        for block in blocks:
            h = block(h)
        return h.reshape((-1, self._num_classes))


class DenseBlockBC(chainer.ChainList):
    def __init__(self, in_channels, num_units, growth_rate=12, drop_rate=0.2):
        '''
        Args:
            in_channels (int):
                Input channels of the block.
            num_units (int):
                Number of primitive functions, i.e. H(x), in the block.
            grouth_rate (int):
                Hyper parameter `k` which is output channels of each H(x).
            drop_rate (int):
                Drop rate for dropout.
        '''

        units = []
        for i in range(num_units):
            units += [BRC1BRC3(in_channels, growth_rate)]
            in_channels = in_channels + growth_rate
        super(DenseBlockBC, self).__init__(*units)
        self.drop_rate = drop_rate

    def __call__(self, x):
        for link in self:
            h = F.dropout(link(x), self.drop_rate)
            x = F.concat((x, h), axis=1)
        return x


class BRC1BRC3(chainer.Chain):
    def __init__(self, in_channels, out_channels, **kwargs):
        bottleneck = 4 * out_channels
        super(BRC1BRC3, self).__init__(
            brc1=BRCChain(in_channels, bottleneck,
                          ksize=1, pad=0, nobias=True),
            brc3=BRCChain(bottleneck, out_channels,
                          ksize=3, pad=1, nobias=True))

    def __call__(self, x):
        h = self.brc1(x)
        y = self.brc3(h)
        return y


class TransitionLayer(chainer.Chain):
    def __init__(self, in_channels, out_channels, global_pool=False):
        super(TransitionLayer, self).__init__(
            brc=BRCChain(in_channels, out_channels, ksize=1))
        self.global_pool = global_pool

    def __call__(self, x):
        h = self.brc(x)
        if self.global_pool:
            ksize = h.shape[2:]
        else:
            ksize = 2
        y = F.average_pooling_2d(h, ksize)
        return y


if __name__ == '__main__':
    # Hyperparameters
    hparams = SimpleNamespace()
    hparams.gpu = 0
    hparams.num_classes = 10
    hparams.nums_units = [16, 16, 16]
    hparams.growth_rate = 24  # out channels of each funcion in dense block
    hparams.dropout_rate = 0.2
    hparams.num_epochs = 300
    hparams.batch_size = 50
    hparams.optimizer = chainer.optimizers.NesterovAG
    hparams.lr_init = 0.1
    hparams.lr_decrease_rate = 0.1
    hparams.weight_decay = 1e-4
    hparams.max_expand_pixel = 8
    hparams.epochs_decrease_lr = [150, 225]

    model = DensenetBC(hparams.num_classes, hparams.nums_units,
                       hparams.growth_rate, hparams.dropout_rate)

    result = common.train_eval(model, hparams)
    best_model, best_test_loss, best_test_acc, best_epoch = result[:4]
    train_loss_log, train_acc_log, test_loss_log, test_acc_log = result[4:]
