import numpy
from chainer import cuda
from chainer import function
from chainer.utils import conv
from chainer.utils import type_check


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


def _matmul(a, b, xp):
    if xp is numpy:
        # numpy 1.9 does not support matmul.
        # So we use numpy.einsum instead of numpy.matmul.
        return xp.einsum('ijk,ikl->ijl', a, b)
    else:
        return xp.matmul(a, b)


class GroupedConvolution2D(function.Function):

    def __init__(self, stride=1, pad=0):
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)

        x_type = in_types[0]
        w_type = in_types[1]
        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == 4,
            w_type.ndim == 4,
            x_type.shape[1] % w_type.shape[1] == 0,
        )

        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward(self, inputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        kh, kw = W.shape[2:]

        xp = cuda.get_array_module(*x)
        if xp is numpy:
            self.col = conv.im2col_cpu(
                x, kh, kw, self.sy, self.sx, self.ph, self.pw)
        else:
            self.col = conv.im2col_gpu(
                x, kh, kw, self.sy, self.sx, self.ph, self.pw)

        B, C, KY, KX, IY, IX = self.col.shape
        D, GC = W.shape[:2]  # (D, GC, KY, KX)
        G = C // GC  # number of gropus
        GD = D // G
        # (B, C, KY, KX, IY, IX) -> (G, B*IY*IX, GC*KY*YX)
        c_ = self.col.reshape((B, G, GC, KY, KX, IY, IX)) \
            .transpose((1, 0, 5, 6, 2, 3, 4)) \
            .reshape((G, B * IY * IX, GC * KY * KX))
        # (G, GC*KY*KX, GD)
        w_ = W.reshape((G, GD, GC * KY * KX)).transpose(0, 2, 1)

        # (G, B*IY*IX, GC*KY*YX), (G, GC*KY*YX, GD)-> (G, B*IY*IX, GD)
        y = _matmul(c_, w_, xp).astype(x.dtype, copy=False)

        # (G, B*IY*IX, GD) -> (B, G*GD, IY, IX)
        y = y.reshape((G, B, IY * IX, GD)).transpose(1, 0, 3, 2) \
            .reshape((B, G * GD, IY, IX))

        if b is not None:
            y += b[None, :, None, None]
        return y,

    def backward(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]
        h, w = x.shape[2:]

        xp = cuda.get_array_module(*x)

        B, C, KY, KX, IY, IX = self.col.shape
        D, GC = W.shape[:2]  # (D, GC, KY, KX)
        G = C // GC  # number of gropus
        GD = D // G

        # (B, G*GD, IY, IX) -> (G, GD, B*IY*IX)
        gy_ = gy.reshape((B, G, GD, IY * IX)).transpose(1, 2, 0, 3) \
            .reshape((G, GD, B * IY * IX))

        # (B, C, KY, KX, IY, IX) -> (G, B*IY*IX, GC*KY*YX)
        c_ = self.col.reshape((B, G, GC, KY, KX, IY, IX)) \
            .transpose((1, 0, 5, 6, 2, 3, 4)) \
            .reshape((G, B * IY * IX, GC * KY * KX))
        # (G, GD, B*IY*IX), (G, B*IY*IX, GC*KY*KX) -> (G, GD, GC*KY*KX)
        gW_ = _matmul(gy_, c_, xp)
        gW = gW_.reshape((D, GC, KY, KX))
        gW = gW.astype(W.dtype, copy=False)

        w_ = W.reshape((G, GD, GC*KY*KX)).transpose(0, 2, 1)
        # (G, GC*KY*KX, GD), (G, GD, B*IY*IX) -> (G, GC*KY*KX, B*IY*IX)
        gcol = _matmul(w_, gy_, xp).reshape((C, KY, KX, B, IY, IX))
        gcol = gcol.astype(x.dtype, copy=False)
        gcol = xp.rollaxis(gcol, 3)

        if xp is numpy:
            gx = conv.col2im_cpu(gcol, self.sy, self.sx,
                                 self.ph, self.pw, h, w)
        else:
            gx = conv.col2im_gpu(gcol, self.sy, self.sx,
                                 self.ph, self.pw, h, w)

        if b is None:
            return gx, gW
        else:
            gy = xp.rollaxis(gy, 1, 4)
            gb = gy.sum(axis=(0, 1, 2))
            return gx, gW, gb


def grouped_convolution_2d(x, W, b=None, stride=1, pad=0):
    """TODO: This document is completely wrong and must be written.
    Two-dimensional depthwise convolution function.

    This is an implementation of two-dimensional depthwise convolution.
    It takes two or three variables: the input image ``x``, the filter weight
    ``W``, and optionally, the bias vector ``b``.

    Notation: here is a notation for dimensionalities.

    - :math:`n` is the batch size.
    - :math:`c_I` is the number of the input.
    - :math:`c_M` is the channel multiplier.
    - :math:`h` and :math:`w` are the height and width of the input image,
      respectively.
    - :math:`h_O` and :math:`w_O` are the height and width of the output image,
      respectively.
    - :math:`k_H` and :math:`k_W` are the height and width of the filters,
      respectively.

    Args:
        x (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
            Input variable of shape :math:`(n, c_I, h, w)`.
        W (~chainer.Variable): Weight variable of shape
            :math:`(c_M, c_I, k_H, k_W)`.
        b (~chainer.Variable):
            Bias variable of length :math:`c_M * c_I` (optional).
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.


    Returns:
        ~chainer.Variable:
            Output variable. Its shape is :math:`(n, c_I * c_M, h_O, w_O)`.

    Like ``Convolution2D``, ``DepthwiseConvolution2D`` function computes
    correlations between filters and patches of size :math:`(k_H, k_W)` in
    ``x``.
    But unlike ``Convolution2D``, ``DepthwiseConvolution2D`` does not add up
    input channels of filters but concatenates them.
    For that reason, the shape of outputs of depthwise convolution are
    :math:`(n, c_I * c_M, h_O, w_O)`, :math:`c_M` is called channel_multiplier.

    :math:`(h_O, w_O)` is determined by the equivalent equation of
    ``Convolution2D``.

    If the bias vector is given, then it is added to all spatial locations of
    the output of convolution.

    See: `L. Sifre. Rigid-motion scattering for image classification\
          <http://www.di.ens.fr/data/publications/papers/phd_sifre.pdf>`_

    .. seealso:: :class:`~chainer.links.DepthwiseConvolution2D`

    .. admonition:: Example

        >>> x = np.random.uniform(0, 1, (2, 3, 4, 7))
        >>> W = np.random.uniform(0, 1, (2, 3, 3, 3))
        >>> b = np.random.uniform(0, 1, (6,))
        >>> y = F.depthwise_convolution_2d(x, W, b)
        >>> y.shape
        (2, 6, 2, 5)

    """
    func = GroupedConvolution2D(stride, pad)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)
