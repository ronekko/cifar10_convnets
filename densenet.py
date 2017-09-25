# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 19:14:24 2017

@author: sakurai

A Chainer implementation of DenseNet,
"Densely Connected Convolutional Networks"
https://arxiv.org/abs/1608.06993v3
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


class Densenet(chainer.ChainList):
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
            funcs.append(DenseBlock(in_channels, num_units, growth_rate,
                                    dropout_rate))
            in_channels += growth_rate * num_units
            out_channels = int(np.ceil(in_channels * compression_factor))
            funcs.append(TransitionLayer(in_channels, out_channels))
        funcs.pop(-1)  # in order to replace the last one with global pooling
        funcs.append(
            TransitionLayer(in_channels, num_classes, global_pool=True))
        super(Densenet, self).__init__(*funcs)
        self._num_classes = num_classes

    def __call__(self, x):
        conv1 = self[0]
        blocks = self[1:]  # dense, transition, ..., dense, transision

        h = conv1(x)
        for block in blocks:
            h = block(h)
        return h.reshape((-1, self._num_classes))


class DenseBlock(chainer.ChainList):
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
        super(DenseBlock, self).__init__(*units)
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
    p = SimpleNamespace()
    p.gpu = 0
    p.num_classes = 10
    p.nums_units = [16, 16, 16]
    p.growth_rate = 12  # out channels of each primitive funcion in dense block
    p.dropout_rate = 0.2
    p.num_epochs = 300
    p.batch_size = 50
    p.lr_init = 0.1
    p.lr_decrease_rate = 0.1
    p.weight_decay = 1e-4
    p.max_expand_pixel = 8
    p.epochs_lr_divide10 = [150, 225]

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
    model = Densenet(p.num_classes, nums_units=p.nums_units,
                     growth_rate=p.growth_rate, dropout_rate=p.dropout_rate)
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
