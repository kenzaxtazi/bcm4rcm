import tqdm
import gpflow
import tensorflow as tf
import numpy as np

from gpflow.base import RegressionData
from gpflow.kernels import Kernel
from gpflow.likelihoods import Likelihood
from gpflow.mean_functions import MeanFunction
from gpflow.models import GPR, SVGP

from typing import List, Optional, Tuple


def get_latent_submodels(
    Xl: List[RegressionData],
    num_inducing_list: List[int],
    kernel: Kernel,
    likelihood: Likelihood = gpflow.likelihoods.Gaussian(variance=0.1),
    mean_function: Optional[MeanFunction] = None,
    ) -> List[SVGP]:
    """
    Helper function to build a list of GPflow SVGP submodels from a list of datasets, a GP prior and a likelihood variance.

    :param maxiter: number of training iterations. If set to -1, no training will occur.

    data is a zipped list of X and Y
    """
 
    if mean_function is None:
        mean_function = gpflow.mean_functions.Zero()

    def _create_submodel(X, M: int) -> SVGP:
    
        num_data = len(X)
        M = min(num_data, M)

        Z = np.linspace(X.min(), X.max(), M)[:, None]  # Z must be of shape [M, 1]

        inducing_variable = gpflow.inducing_variables.SeparateIndependentInducingVariables(
            [
                gpflow.inducing_variables.InducingPoints(Z),  # This is U1 = f1(Z1)
                gpflow.inducing_variables.InducingPoints(Z),  # This is U2 = f2(Z2)
            ])
        
        #gpflow.set_trainable(inducing_variable, False)

        submodel = gpflow.models.SVGP(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            mean_function=mean_function,
            num_latent_gps=likelihood.latent_dim)
        
        gpflow.utilities.set_trainable(submodel.q_mu, False)
        gpflow.utilities.set_trainable(submodel.q_sqrt, False)

        return submodel
    
    models = [
        _create_submodel(X, M)
        for X, M in tqdm.tqdm(
            zip(Xl, num_inducing_list), total=len(Xl))]
    return models



class OLD_FixedVarianceLikelihood(gpflow.functions.Function):
    """ Fixed Variance likelihood function"""
    def __init__(self, Y: gpflow.base.AnyNDArray):
        self.var_mean = np.var(Y, axis=-1, keepdims=True)

    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        return self.var_mean