# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 12:46:40 2017

@author: sakurai

A Chainer implementation of ResNeXt,
"Aggregated Residual Transformations for Deep Neural Networks",
https://arxiv.org/abs/1611.05431v2
"""

from types import SimpleNamespace

import chainer
import chainer.functions as F
import chainer.links as L

import common
from functions import extend_channels
from links import BRCChain, GroupedConvolution2D


class Resnext(chainer.Chain):
    '''
    Args:
        n_blocks (int):
            Number of blocks in each stage.
        cardinality (int):
            Number of sub-layers in grouped conv layers.
    '''
    def __init__(self, n_blocks, channels, ch_bottleneck,
                 cardinality, ch_group_out):
        super(Resnext, self).__init__(
            conv1=L.Convolution2D(3, 64, 3, pad=1),  # appendix A
            stage2=ResnextStage(
                n_blocks, channels, ch_bottleneck, cardinality, ch_group_out),
            stage3=ResnextStage(
                n_blocks, channels, ch_bottleneck, cardinality, ch_group_out),
            stage4=ResnextStage(
                n_blocks, channels, ch_bottleneck, cardinality, ch_group_out),
            brc_out=BRCChain(channels, 10, 1, pad=0)
        )

    def __call__(self, x):
        h = self.conv1(x)
        h = self.stage2(h)
        h = F.max_pooling_2d(h, 2)
        h = self.stage3(h)
        h = F.max_pooling_2d(h, 2)
        h = self.stage4(h)
        h = self.brc_out(h)
        y = F.average_pooling_2d(h, h.shape[2:]).reshape(-1, 10)
        return y


class ResnextStage(chainer.ChainList):
    '''Sequence of `ResnextBlock` s.
    '''
    def __init__(self, n_blocks, channels, ch_bottleneck,
                 cardinality, ch_group_out):
        blocks = [
            ResnxetBlock(channels, ch_bottleneck, cardinality, ch_group_out)
            for i in range(n_blocks)]
        super(ResnextStage, self).__init__(*blocks)
        self._channels = channels

    def __call__(self, x):
        x = extend_channels(x, self._channels)
        for link in self:
            x = link(x)
        return x


class ResnxetBlock(chainer.Chain):
    '''
    Args:
        ch_in (int):
            Number of channels of input to the block.
        ch_bottleneck (int):
            Number of channels of output from the first conv (or equevalently
            input to the grouped conv).
        cardinality (int):
            Number of groups (i.e. paths) in the grouped conv.
        ch_group_out (int):
            Number of channels of output for each group in the grouped conv.
            Note that thus the output channels of the grouped conv layer is
            `cardinality * ch_group_out` .
        ch_out (int, optional):
            Number of output channels of this block if `ch_out` is specified,
            otherwise `ch_out` is the same as `ch_int` .
    '''
    def __init__(self, ch_in, ch_bottleneck=64,
                 cardinality=4, ch_group_out=64, ch_out=None):
        if ch_out is None:
            ch_out = ch_in
        ch_gconv_out = cardinality * ch_group_out
        super(ResnxetBlock, self).__init__(
            brc1=BRCChain(ch_in, ch_bottleneck, 1),
            brg2=BRGChain(cardinality, ch_bottleneck, ch_gconv_out, 3, pad=1),
            brc3=BRCChain(ch_gconv_out, ch_out, 1))

    def __call__(self, x):
        h = self.brc1(x)
        h = self.brg2(h)
        h = self.brc3(h)
        return x + h


class BRGChain(chainer.Chain):
    '''
    This is a composite link of sequence of BatchNormalization, ReLU and
    GroupedConvolution2D.
    '''
    def __init__(
            self, cardinality, in_channels, out_channels, ksize, stride=1,
            pad=0, nobias=False, initialW=None, initial_bias=None, **kwargs):
        in_ch, out_ch = in_channels, out_channels
        super(BRGChain, self).__init__(
            bn=L.BatchNormalization(in_ch),
            gconv=GroupedConvolution2D(
                cardinality, in_ch, out_ch, ksize=ksize, stride=stride,
                pad=pad, nobias=nobias, initialW=initialW,
                initial_bias=initial_bias, **kwargs))

    def __call__(self, x):
        h = self.bn(x)
        h = F.relu(h)
        y = self.gconv(h)
        return y


if __name__ == '__main__':
    # Hyperparameters
    hparams = SimpleNamespace()
    hparams.gpu = 0                # GPU>=0, CPU < 0
    hparams.n_blocks = 3   # number of blocks in each stage
    hparams.channels = 256
    hparams.ch_bottleneck = 128  # 64
    hparams.cardinality = 32  # 4
    hparams.ch_group_out = 4  # 64
    hparams.num_epochs = 300  # appendix A
    hparams.batch_size = 100
    hparams.optimizer = chainer.optimizers.NesterovAG
    hparams.lr_init = 0.1  # appendix A
    hparams.lr_decrease_rate = 0.1  # appendix A
    hparams.weight_decay = 5e-4  # appendix A 5e-4
    hparams.epochs_decrease_lr = [150, 225]  # appendix A

    model = Resnext(hparams.n_blocks, hparams.channels, hparams.ch_bottleneck,
                    hparams.cardinality, hparams.ch_group_out)

    result = common.train_eval(model, hparams)
    best_model, best_test_loss, best_test_acc, best_epoch = result[:4]
    train_loss_log, train_acc_log, test_loss_log, test_acc_log = result[4:]
