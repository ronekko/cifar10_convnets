import unittest

import numpy

from chainer import cuda
from chainer.functions.connection import convolution_2d
import grouped_convolution_2d
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*(testing.product({
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
})))
class TestGroupedConvolution2DFunction(unittest.TestCase):

    def setUp(self):
        B = 3  # batch size
        C = 6  # number of channels of input tensor
        D = 6  # number of channels of output tensor
        self.group = 2
        GC = C // self.group
        ih, iw = (5, 5)
        kh, kw = (3, 3)
        oh, ow = ih - kh + 1, iw - kw + 1
        self.stride = 2
        self.pad = 1
        self.W = numpy.random.normal(
            0, numpy.sqrt(1. / (kh * kw * GC)),
            (D, GC, kh, kw)).astype(self.W_dtype)
        self.b = numpy.random.uniform(
            -1, 1, D).astype(self.x_dtype)

        self.x = numpy.random.uniform(
            -1, 1, (B, C, ih, iw)).astype(self.x_dtype)
        self.gy = numpy.random.uniform(
            -1, 1, (B, D, oh, ow)).astype(self.x_dtype)
        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.x_dtype == numpy.float16 or self.W_dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-3, 'rtol': 5e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-3, 'rtol': 5e-3}

    def check_forward(self, x_data, W_data, b_data):
        xp = cuda.get_array_module(x_data)
        args1 = (x_data, W_data)
        if b_data is not None:
            args1 = args1 + (b_data,)

        f1 = grouped_convolution_2d.GroupedConvolution2D(self.stride, self.pad)
        y1 = f1(*args1).data

        f2 = convolution_2d.Convolution2DFunction(self.stride, self.pad)
        x_groups = xp.split(x_data, self.group, axis=1)
        W_groups = xp.split(W_data, self.group, axis=0)
        if b_data is not None:
            b_groups = xp.split(b_data, self.group, axis=0)
        y2s = []
        for g in range(len(x_groups)):
            args2 = (x_groups[g], W_groups[g])
            if b_data is not None:
                args2 = args2 + (b_groups[g],)
            y2s.append(f2(*args2).data)
        y2 = xp.concatenate(y2s, axis=1)
        testing.assert_allclose(y1, y2, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.W, self.b)

    def test_forward_cpu_nobias(self):
        self.check_forward(self.x, self.W, None)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                           cuda.to_gpu(self.b))

    @attr.gpu
    def test_forward_gpu_nobias(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.W), None)

    def check_backward(self, x_data, W_data, b_data, y_grad):
        args = (x_data, W_data)
        if b_data is not None:
            args = args + (b_data,)

        gradient_check.check_backward(
            grouped_convolution_2d.GroupedConvolution2D(self.stride, self.pad),
            args, y_grad, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.b, self.gy)

    @condition.retry(3)
    def test_backward_cpu_nobias(self):
        self.check_backward(self.x, self.W, None, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            cuda.to_gpu(self.b), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_nobias(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            None, cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
