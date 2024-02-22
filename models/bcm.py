import gpflow
import tensorflow as tf
import numpy as np


class FixedVarianceLikelihood(gpflow.functions.Function):
    """ Fixed Variance likelihood function"""
    def __init__(self, Y: gpflow.base.AnyNDArray):
        self.var_mean = np.var(Y, axis=-1, keepdims=True)

    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        return self.var_mean