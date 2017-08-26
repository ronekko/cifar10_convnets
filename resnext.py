# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 12:46:40 2017

@author: sakurai

A Chainer implementation of ResNeXt,
"Aggregated Residual Transformations for Deep Neural Networks",
https://arxiv.org/abs/1611.05431v2
"""

from copy import deepcopy
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from datasets import (
    load_cifar10_as_ndarray, random_augment, random_augment2)

from links import BRCChain, GroupedConvolution2D
from functions import extend_channels


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
    p = SimpleNamespace()
    p.gpu = 0                # GPU>=0, CPU < 0
    p.n_blocks = 3   # number of blocks in each stage
    p.channels = 256
    p.ch_bottleneck = 128  # 64
    p.cardinality = 32  # 4
    p.ch_group_out = 4  # 64
    p.num_epochs = 300  # appendix A
    p.batch_size = 100
    p.lr_init = 0.1  # appendix A
    p.lr_decrease_rate = 0.1  # appendix A
    p.weight_decay = 5e-4  # appendix A 5e-4
    p.epochs_lr_divide10 = [150, 225]  # appendix A

    xp = np if p.gpu < 0 else chainer.cuda.cupy

    # Dataset
    train, test = load_cifar10_as_ndarray(3)
    x_train, c_train = train
    x_test, c_test = test
    num_train = len(x_train)
    num_test = len(x_test)
    std_rgb = x_train.std((0, 2, 3), keepdims=True)
    x_train /= std_rgb
    x_test /= std_rgb
    mean_rgb = x_train.mean((0, 2, 3), keepdims=True)
    x_train -= mean_rgb
    x_test -= mean_rgb

    # Model and optimizer
    model = Resnext(p.n_blocks, p.channels, p.ch_bottleneck,
                    p.cardinality, p.ch_group_out)
    if p.gpu >= 0:
        model.to_gpu()
    optimizer = optimizers.NesterovAG(p.lr_init)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(p.weight_decay))

    # Training loop
    train_loss_log = []
    train_acc_log = []
    test_loss_log = []
    test_acc_log = []
    best_test_acc = 0
    try:
        for epoch in range(p.num_epochs):
            if epoch in p.epochs_lr_divide10:
                optimizer.lr *= p.lr_decrease_rate

            epoch_losses = []
            epoch_accs = []
            for i in tqdm(range(0, num_train, p.batch_size)):
#                x_batch = random_augment(x_train[i:i+batch_size],
#                                         max_expand_pixel)
                x_batch = random_augment2(x_train[i:i+p.batch_size])
                x_batch = xp.asarray(x_batch)
                c_batch = xp.asarray(c_train[i:i+p.batch_size])
                model.cleargrads()
                with chainer.using_config('train', True):
                    y_batch = model(x_batch)
                    loss = F.softmax_cross_entropy(y_batch, c_batch)
                    acc = F.accuracy(y_batch, c_batch)
                    loss.backward()
                optimizer.update()
                epoch_losses.append(loss.data)
                epoch_accs.append(acc.data)

            epoch_loss = np.mean(chainer.cuda.to_cpu(xp.stack(epoch_losses)))
            epoch_acc = np.mean(chainer.cuda.to_cpu(xp.stack(epoch_accs)))
            train_loss_log.append(epoch_loss)
            train_acc_log.append(epoch_acc)

            # Evaluate the test set
            losses = []
            accs = []
            for i in tqdm(range(0, num_test, p.batch_size)):
                x_batch = xp.asarray(x_test[i:i+p.batch_size])
                c_batch = xp.asarray(c_test[i:i+p.batch_size])
                with chainer.using_config('train', False):
                    y_batch = model(x_batch)
                    loss = F.softmax_cross_entropy(y_batch, c_batch)
                    acc = F.accuracy(y_batch, c_batch)
                losses.append(loss.data)
                accs.append(acc.data)
            test_loss = np.mean(chainer.cuda.to_cpu(xp.stack(losses)))
            test_acc = np.mean(chainer.cuda.to_cpu(xp.stack(accs)))
            test_loss_log.append(test_loss)
            test_acc_log.append(test_acc)

            # Keep the best model so far
            if test_acc > best_test_acc:
                best_model = deepcopy(model)
                best_test_loss = test_loss
                best_test_acc = test_acc
                best_epoch = epoch

            # Display the training log
            print('{}: loss = {}'.format(epoch, epoch_loss))
            print('test acc = {}'.format(test_acc))
            print('best test acc = {} (# {})'.format(best_test_acc,
                                                     best_epoch))
            print(p)

            plt.figure(figsize=(10, 4))
            plt.title('Loss')
            plt.plot(train_loss_log, label='train loss')
            plt.plot(test_loss_log, label='test loss')
            plt.legend()
            plt.grid()
            plt.show()

            plt.figure(figsize=(10, 4))
            plt.title('Accucary')
            plt.plot(train_acc_log, label='train acc')
            plt.plot(test_acc_log, label='test acc')
            plt.legend()
            plt.grid()
            plt.show()

    except KeyboardInterrupt:
        print('Interrupted by Ctrl+c!')

    print('best test acc = {} (# {})'.format(best_test_acc,
                                             best_epoch))
    print(p)
