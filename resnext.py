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
from links import BRCChain


class Resnext(chainer.Chain):
    '''
    Args:
        n (int):
            Number of blocks in each group.
    '''
    def __init__(self, cardinality=8, ch_first_conv=64, num_blocks=[3, 3, 3],
                 ch_blocks=[64, 128, 256]):
        ch_blocks = [2 * ch * cardinality for ch in ch_blocks]

        n = num_blocks
        ch = ch_blocks
        super(Resnext, self).__init__(
            conv1=L.Convolution2D(3, ch_first_conv, ksize=3, pad=1),
            stage2=ResnextStage(n[0], ch[0], cardinality, False),
            stage3=ResnextStage(n[1], ch[1], cardinality, True),
            stage4=ResnextStage(n[2], ch[2], cardinality, True),
            bn_out=L.BatchNormalization(ch[2]),
            fc_out=L.Linear(ch[2], 10)
        )
        self.first_stage_in_ch = ch[0]

    def __call__(self, x):
        h = self.conv1(x)
        h = extend_channels(h, self.first_stage_in_ch)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        h = self.bn_out(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, h.shape[2:])
        y = self.fc_out(h)
        return y


class ResnextStage(chainer.ChainList):
    '''Sequence of `ResnetBlock`s.
    '''
    def __init__(self, n_blocks, channels, cardinality, transition):
        n_blocks = n_blocks - 1
        blocks = [ResnextBlock(channels, cardinality, transition)]
        blocks += [ResnextBlock(channels, cardinality)
                   for i in range(n_blocks)]
        super(ResnextStage, self).__init__(*blocks)

    def __call__(self, x):
        for block in self:
            x = block(x)
        return x


class ResnextBlock(chainer.Chain):
    '''
    Args:
        ch_out (int):
            Number of channels of output of the block.
        cardinality (int):
            Number of groups (i.e. paths) in the grouped conv.
    '''
    def __init__(self, ch_out, cardinality, transition=False):
        self.transition = transition
        ch_in = ch_out // 2 if transition else ch_out
        stride = 2 if transition else 1
        bottleneck = ch_out // 2
        super(ResnextBlock, self).__init__(
            brc1=BRCChain(ch_in, bottleneck, 1, stride, pad=0, nobias=True),
            brg2=BRGChain(bottleneck, bottleneck, 3, pad=1, group=cardinality),
            brc3=BRCChain(bottleneck, ch_out, 1, pad=0, nobias=True))

    def __call__(self, x):
        h = self.brc1(x)
        h = self.brg2(h)
        h = self.brc3(h)
        if self.transition:
            x = avgpool_and_extend_channels(x)
        return x + h


def avgpool_and_extend_channels(x, ch_out=None, ksize=2):
    ch_in = x.shape[1]
    if ch_out is None:
        ch_out = ch_in * 2
    x = F.average_pooling_2d(x, ksize)
    return extend_channels(x, ch_out)


class BRGChain(chainer.Chain):
    '''
    This is a composite link of sequence of BatchNormalization, ReLU and
    GroupedConvolution2D.
    '''
    def __init__(
            self, in_channels, out_channels, ksize, stride=1, pad=0,
            nobias=False, initialW=None, initial_bias=None, group=1, **kwargs):
        in_ch, out_ch = in_channels, out_channels
        super(BRGChain, self).__init__(
            bn=L.BatchNormalization(in_ch),
            gconv=L.Convolution2D(
                in_ch, out_ch, ksize=ksize, stride=stride,
                pad=pad, nobias=nobias, initialW=initialW,
                initial_bias=initial_bias, group=group, **kwargs))

    def __call__(self, x):
        h = self.bn(x)
        h = F.relu(h)
        y = self.gconv(h)
        return y


if __name__ == '__main__':
    # Hyperparameters
    hparams = SimpleNamespace()
    hparams.gpu = 0                # GPU>=0, CPU < 0

    # Parameters for network
    hparams.cardinality = 8
    hparams.ch_first_conv = 64
    hparams.num_blocks = [3, 3, 3]
    hparams.ch_blocks = [16, 32, 64]

    # Parameters for optimization
    hparams.num_epochs = 300  # appendix A
    hparams.batch_size = 100
    hparams.optimizer = chainer.optimizers.NesterovAG
    hparams.lr_init = 0.1  # appendix A
    hparams.lr_decrease_rate = 0.1  # appendix A
    hparams.weight_decay = 5e-4  # appendix A 5e-4
    hparams.epochs_decrease_lr = [150, 225]  # appendix A

    model = Resnext(hparams.cardinality, hparams.ch_first_conv,
                    hparams.num_blocks, hparams.ch_blocks)
    result = common.train_eval(model, hparams)
    best_model, best_test_loss, best_test_acc, best_epoch = result[:4]
    train_loss_log, train_acc_log, test_loss_log, test_acc_log = result[4:]
