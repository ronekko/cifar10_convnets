# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 17:33:56 2017

@author: sakurai
"""

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


class BRCChain(chainer.Chain):
    '''
    This is a composite link of sequence of BatchNormalization, ReLU and
    Convolution2D (a.k.a. pre-activation unit).
    '''
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, decay=0.9,
                 **kwargs):
        in_ch, out_ch = in_channels, out_channels
        super(BRCChain, self).__init__(
            bn=L.BatchNormalization(in_ch, decay=decay),
            conv=L.Convolution2D(in_ch, out_ch, ksize=ksize, stride=stride,
                                 pad=pad, nobias=nobias, initialW=initialW,
                                 initial_bias=initial_bias, **kwargs))

    def __call__(self, x):
        h = self.bn(x)
        h = F.relu(h)
        y = self.conv(h)
        return y


class BRPChain(chainer.Chain):
    '''
    This is a composite link of sequence of BatchNormalization, ReLU and
    global AveragePooling2D.
    '''
    def __init__(self, in_channels):
        super(BRPChain, self).__init__(
            bn=L.BatchNormalization(in_channels))

    def __call__(self, x):
        h = self.bn(x)
        h = F.relu(h)
        y = F.average_pooling_2d(h, h.shape[2:])
        return y


class SeparableConvolution2D(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, **kwargs):
        if 'pad' in kwargs:
            pad = kwargs.pop('pad')
        else:
            pad = 0
        super(SeparableConvolution2D, self).__init__(
            depthwise=L.DepthwiseConvolution2D(in_channels, 1, ksize, pad=pad,
                                               **kwargs),
            pointwise=L.Convolution2D(in_channels, out_channels, 1, **kwargs)
        )

    def __call__(self, x):
        return self.pointwise(self.depthwise(x))
