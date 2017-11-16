# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:37:15 2017

@author: sakurai

Implementation of
"The Reversible Residual Network: Backpropagation Without Storing Activations".
"""

from types import SimpleNamespace

import chainer
import chainer.functions as F
import chainer.links as L

import common
from links import BRCChain


class Revnet(chainer.Chain):
    '''
    Reversible Residual Network.

    Args:
        n (int):
            Number of units in each group.
    '''

    def __init__(self, n=6, channels=[32, 32, 64, 112], use_bottleneck=False):
        if not use_bottleneck:  # default case
            ch_out = channels
        else:
            ch_out = [channels[0]] + [ch * 4 for ch in channels[1:]]

        super(Revnet, self).__init__(
            conv1=L.Convolution2D(3, ch_out[0], 3, pad=1, nobias=True),
            stage2=RevnetStage(n, ch_out[1], use_bottleneck),
            stage3=RevnetStage(n, ch_out[2], use_bottleneck),
            stage4=RevnetStage(n, ch_out[3], use_bottleneck),
            bn_out=L.BatchNormalization(ch_out[3]),
            fc_out=L.Linear(ch_out[3], 10)
        )

    def __call__(self, x):
        h = self.conv1(x)
        h = self.stage2(h)
        h = F.max_pooling_2d(h, 2)
        h = self.stage3(h)
        h = F.max_pooling_2d(h, 2)
        h = self.stage4(h)
        h = self.bn_out(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, h.shape[2:])
        y = self.fc_out(h)
        return y


class RevnetStage(chainer.ChainList):
    '''Reversible sequence of `ResnetUnit`s.
    '''
    def __init__(self, n_blocks, channels, use_bottleneck=True):
        if use_bottleneck:
            unit_class = RevnetBottleneckUnit
        else:
            unit_class = RevnetUnit
        blocks = [unit_class(channels // 2) for i in range(n_blocks)]
        super(RevnetStage, self).__init__(*blocks)
        self._channels = channels

    def __call__(self, x):
        x = extend_channels(x, self._channels)
        revnet_stage_function = RevnetStageFunction(self)
        y = revnet_stage_function(x)
        return y


class RevnetStageFunction(chainer.Function):
    def __init__(self, chainlist):
        """
        Args:
            chainlist (chainer.Chainlist):
                A ChainList of revnet units.
        """
        self.chainlist = chainlist

    def forward(self, inputs):
        xp = chainer.cuda.get_array_module(*inputs)
        x = inputs[0]
        x1, x2 = xp.split(x, 2, axis=1)

        with chainer.no_backprop_mode():
            x1 = chainer.Variable(x1)
            x2 = chainer.Variable(x2)
            for res_unit in self.chainlist:
                x2 += res_unit(x1)
                x1, x2 = x2, x1

        y = xp.concatenate((x1.array, x2.array), axis=1)
        self.retain_outputs((0,))
        return y,

    def backward(self, inputs, grads):
        xp = chainer.cuda.get_array_module(*grads)
        y_array = self.output_data[0]
        grad_y = grads[0]

        y1_array, y2_array = xp.split(y_array, 2, axis=1)
        grad_y1, grad_y2 = xp.split(grad_y, 2, axis=1)

        a, b = y1_array.copy(), y2_array.copy()
        ga, gb = grad_y1.copy(), grad_y2.copy()
        for res_unit in self.chainlist[::-1]:
            b_var = chainer.Variable(b)
            with chainer.force_backprop_mode():
                c_var = res_unit(b_var)
                c_var.grad = ga
                c_var.backward()
            a -= c_var.array
            gb += b_var.grad
            a, b = b, a
            ga, gb = gb, ga

        gx = xp.concatenate((ga, gb), axis=1)
        return gx,


class RevnetUnit(chainer.Chain):
    '''The function F or G in the revnet paper.
    '''
    def __init__(self, channels):
        super(RevnetUnit, self).__init__(
            # In revnet training, BN's `decay` parameters should be `sqrt`ed
            # in order to compensate double forward passes for one update.
            brc1=BRCChain(channels, channels, 3, pad=1, decay=0.9**0.5),
            brc2=BRCChain(channels, channels, 3, pad=1, decay=0.9**0.5))

    def __call__(self, x):
        h = self.brc1(x)
        h = self.brc2(h)
        return h


class RevnetBottleneckUnit(chainer.Chain):
    '''The function F or G in the revnet paper.
    '''
    def __init__(self, channels):
        bottleneck = channels // 4
        super(RevnetBottleneckUnit, self).__init__(
            # In revnet training, BN's `decay` parameters should be `sqrt`ed
            # in order to compensate double forward passes for one update.
            brc1=BRCChain(channels, bottleneck, 1, pad=0, decay=0.9**0.5),
            brc2=BRCChain(bottleneck, bottleneck, 3, pad=1, decay=0.9**0.5),
            brc3=BRCChain(bottleneck, channels, 1, pad=0, decay=0.9**0.5))

    def __call__(self, x):
        h = self.brc1(x)
        h = self.brc2(h)
        h = self.brc3(h)
        return h


def extend_channels(x, out_ch):
    '''Extends channels (i.e. depth) of the input BCHW tensor x by zero-padding
    if out_ch is larger than the number of channels of x, otherwise returns x.

    Note that this function is different from `functions.extend_channels` that
    pads a zero-filled tensor by concatenating it to the end of `x`
    as following:
        [1, 2, 3, 4] -> [1, 2, 3, 4, 0, 0, 0, 0, 0, 0]
    On the other hand, this function is modified to fit to use with revnet that
    pads zeros as following:
        [1, 2, 3, 4] -> [1, 2, 0, 0, 0, 3, 4, 0, 0, 0]
    '''
    b, in_ch, h, w = x.shape
    if in_ch == out_ch:
        return x
    elif in_ch > out_ch:
        raise ValueError('out_ch must be larger than x.shape[1].')

    xp = chainer.cuda.get_array_module(x)
    x1, x2 = F.split_axis(x, 2, axis=1)
    filler_shape = (b, (out_ch - in_ch) // 2, h, w)
    filler = xp.zeros(filler_shape, x.dtype)
    return F.concat((x1, filler, x2, filler), axis=1)


if __name__ == '__main__':
    # Hyperparameters
    hparams = SimpleNamespace()
    hparams.gpu = 0  # GPU>=0, CPU < 0
    hparams.use_bottleneck = True
    hparams.n = 18   # number of units in each stage
    hparams.channels = [32, 32, 64, 128]
    hparams.num_epochs = 160
    hparams.batch_size = 100
    hparams.lr_init = 0.1
    hparams.lr_decrease_rate = 0.1
    hparams.weight_decay = 2e-4
    hparams.epochs_lr_divide10 = [80, 120]

    model = Revnet(hparams.n, hparams.channels, hparams.use_bottleneck)

    result = common.train_eval(model, hparams)
    best_model, best_test_loss, best_test_acc, best_epoch = result[:4]
    train_loss_log, train_acc_log, test_loss_log, test_acc_log = result[4:]
