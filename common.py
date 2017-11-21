# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:52:19 2017

@author: sakurai
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import chainer
import chainer.functions as F

from datasets import load_cifar10_as_ndarray, random_augment_padding


def train_eval(model, hparams):
    p = hparams
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
    if p.gpu >= 0:
        model.to_gpu()
    optimizer = p.optimizer(p.lr_init)
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
            if epoch in p.epochs_decrease_lr:
                optimizer.lr *= p.lr_decrease_rate

            epoch_losses = []
            epoch_accs = []
            perm = np.random.permutation(num_train)
            index_batches = np.split(perm, num_train // p.batch_size)
            for i_batch in tqdm(index_batches):
                x_batch = random_augment_padding(x_train[i_batch])
                x_batch = xp.asarray(x_batch)
                c_batch = xp.asarray(c_train[i_batch])
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
            plt.ylim(0, 1)
            plt.legend()
            plt.grid()
            plt.show()

            plt.figure(figsize=(10, 4))
            plt.title('Accucary')
            plt.plot(train_acc_log, label='train acc')
            plt.plot(test_acc_log, label='test acc')
            plt.ylim(0.6, 1)
            plt.legend()
            plt.grid()
            plt.show()

    except KeyboardInterrupt:
        print('Interrupted by Ctrl+c!')

    print('best test acc = {} (# {})'.format(best_test_acc,
                                             best_epoch))
    print(p)
    print()

    best_model.cleargrads()
    return (best_model, best_test_loss, best_test_acc, best_epoch,
            train_loss_log, train_acc_log, test_loss_log, test_acc_log)
