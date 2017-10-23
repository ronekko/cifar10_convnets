# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 17:33:56 2017

@author: sakurai
"""

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

import grouped_convolution_2d

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


class GroupedConvolution2D(chainer.Link):

    """Two-dimensional grouped convolutional layer, which is used in ResNeXt.

    This link wraps the :func:`~chainer.functions.grouped_convolution_2d`
    function and holds the filter weight and bias vector as parameters.

    Args:
        n_groups (int): Number of groups into which the channel axis of input
            array is divided. Both ``in_channels`` and ``out_channels`` must be
            divisible by ``group`` .
        in_channels (int): Number of channels of input arrays. If ``None``,
            parameter initialization will be deferred until the first forward
            data pass at which time the size will be determined.
             ``in_channels`` must be divisible by  ``n_groups`` .
        out_channels (int): Number of channels of output arrays.
             ``out_channels`` must be divisible by  ``n_groups`` .

        ksize (int or pair of ints): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k)`` are equivalent.
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        nobias (bool): If ``True``, then this link does not use the bias term.
        initialW (4-D array): Initial weight value. If ``None``, the default
            initializer is used.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        initial_bias (1-D array): Initial bias value. If ``None``, the bias
            is set to 0.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.

    .. seealso::
       See :func:`chainer.functions.grouped_convolution_2d`.

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.

    """

    def __init__(self, n_groups, in_channels, out_channels, ksize, stride=1,
                 pad=0, nobias=False, initialW=None, initial_bias=None):

        if in_channels % n_groups != 0:
            raise ValueError('in_channels must be divisible by n_groups')
        if out_channels % n_groups != 0:
            raise ValueError('out_channels must be divisible by n_groups')

        super(GroupedConvolution2D, self).__init__()
        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.n_groups = n_groups
        self.nobias = nobias

        if initialW is None:
            initialW = chainer.initializers.HeNormal(1. / np.sqrt(2))

        with self.init_scope():
            W_initializer = chainer.initializers._get_initializer(initialW)
            self.W = chainer.Parameter(W_initializer)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = chainer.initializers.Constant(0)
                bias_initializer = chainer.initializers._get_initializer(
                    initial_bias)
                self.b = chainer.Parameter(bias_initializer)

        if in_channels is not None:
            self._initialize_params(in_channels, out_channels)

    def _initialize_params(self, in_channels, out_channels):
        kh, kw = _pair(self.ksize)
        W_shape = (out_channels, in_channels // self.n_groups, kh, kw)
        self.W.initialize(W_shape)
        if self.b is not None:
            self.b.initialize(out_channels)

    def __call__(self, x):
        """Applies the depthwise convolution layer.

        Args:
            x (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
                Input image.

        Returns:
            ~chainer.Variable: Output of the depthwise convolution.

        """
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return grouped_convolution_2d.grouped_convolution_2d(
            x, self.W, self.b, self.stride, self.pad)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
