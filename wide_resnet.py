# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 16:06:25 2017

@author: sakurai

Implementation of "Wide Residual Networks".
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

from links import BRCChain
from functions import extend_channels


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
    p = SimpleNamespace()
    p.gpu = -1  # GPU>=0, CPU < 0
    p.n = 4   # number of blocks in each group
    p.k = 10  # widening factor
    p.num_epochs = 200
    p.batch_size = 100
    p.lr_init = 0.1
    p.lr_decrease_rate = 0.2
    p.weight_decay = 5e-4
    p.epochs_lr_divide10 = [60, 120, 160]

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
    model = WideResnet(n=p.n, k=p.k)
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
