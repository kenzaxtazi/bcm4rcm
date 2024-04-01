import abc
import tqdm
import gpflow
import tensorflow as tf
import numpy as np

from gpflow.kernels import Kernel
from gpflow.models import SVGP, GPModel
from gpflow.likelihoods import Likelihood
from gpflow.base import InputData, MeanAndVariance, RegressionData


from enum import Enum
from itertools import zip_longest
from typing import List, Optional, Union

from check_shapes import check_shape as cs
from check_shapes import check_shapes


class EnsembleMethods(Enum):
    """Aggregation methods"""

    POE = "PoE"
    GPOE = "gPoE"
    BCM = "BCM"
    RBCM = "rBCM"
    BARY = "Bary"
    MoE = "MoE"

class WeightingMethods(Enum):
    """Weighting methods"""

    VAR = "Var"
    WASS = "Wasser"
    UNI = "Uniform"
    ENT = "Entropy"
    NONE = "NoWeights"
    KL = "KL"


def compute_weights(
    mu_s: tf.Tensor,
    var_s: tf.Tensor,
    power: float,
    weighting: WeightingMethods,
    prior_var: Optional[tf.Tensor] = None,
    softmax: bool = False,
) -> tf.Tensor:

    """Compute unnormalized weight matrix

    Inputs :
        - mu_s: predictive mean of each expert (P) at each test point (N) for each output (L)
        - var_s: predictive (marginal) variance of each expert (P) at each test point (N) for each output (L)
        - var_s: dimension: n_expert x n_test_points : predictive variance of each expert at each test point
        - power: scalar, Softmax scaling
        - weighting: weighting method
        - prior_var, prior variance
        - soft_max_wass : whether to use softmax scaling or fraction scaling

    Output :
        -- weight_matrix, dimension: n_expert x n_test_points : unnormalized weight of ith expert at jth test point
    """

    if weighting == WeightingMethods.VAR:
        return tf.math.exp(-power * var_s)

    elif weighting == WeightingMethods.WASS:
        wass = mu_s**2 + (var_s - prior_var) #** 2
        if softmax:
            return tf.math.exp(power * wass)
        else:
            return wass**power

    elif weighting == WeightingMethods.UNI:
        num_experts = tf.cast(tf.shape(mu_s)[-1], mu_s.dtype)
        return tf.ones_like(mu_s) / num_experts

    elif weighting == WeightingMethods.ENT:
        return 0.5 * (tf.math.log(prior_var) - tf.math.log(var_s))

    elif weighting == WeightingMethods.NONE:
        return tf.ones_like(mu_s)

    else:
        raise NotImplementedError("Unknown weighting passed to compute_weights.")


def get_latent_submodels(
    Xl: np.ndarray,
    inducing_pts: np.ndarray,
    kernel: Kernel,
    likelihood: Likelihood = gpflow.likelihoods.Gaussian(variance=0.1),
    ) -> List[SVGP]:
    """
    Helper function to build a list of GPflow SVGP submodels from a list of datasets, a GP prior and a likelihood variance.

    :param maxiter: number of training iterations. If set to -1, no training will occur.

    data is a zipped list of X and Y
    """
 
    mean_function = gpflow.mean_functions.Zero()

    def _create_submodel(X:np.ndarray, Z:np.ndarray) -> SVGP:

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
        _create_submodel(X, Z)
        for X, Z in tqdm.tqdm(
            zip(Xl, inducing_pts), total=len(Xl))]
    return models

class latent_GPEnsemble(GPModel, metaclass=abc.ABCMeta):
    """
    Base class for latent GP ensembles.
    """

    def __init__(
        self,
        models: List[GPModel],
    ):
        """
        :param models: A list of GPflow models with the same prior and likelihood.
        """
        # check that all models are of the same type (e.g., GPR, SVGP)
        # check that all models have the same prior
        for model in models[1:]:
            assert (
                model.kernel == models[0].kernel
            ), "All submodels must have the same kernel"
            assert (
                model.likelihood == models[0].likelihood
            ), "All submodels must have the same likelihood"
            assert (
                model.mean_function == models[0].mean_function
            ), "All submodels must have the same mean function"
            assert (
                model.num_latent_gps == models[0].num_latent_gps
            ), "All submodels must have the same number of latent GPs"

        GPModel.__init__(
            self,
            kernel=models[0].kernel,
            likelihood=models[0].likelihood,
            mean_function=models[0].mean_function,
            num_latent_gps=models[0].num_latent_gps,
        )

        self.models: List[GPModel] = models

    @property
    def trainable_variables(self):  # type: ignore
        r = []
        for model in self.models:
            r += model.trainable_variables
        return r
    
    @property
    def variational_variables(self): # type: ignore
        r = []
        for model in self.models:
            r.append([model.q_mu, model.q_sqrt])
        return r
    
    def maximum_log_likelihood_objective(self, data: List[RegressionData]) -> tf.Tensor:  # type: ignore
        """ This give a scalar value, data is a zipped list of X and Y """
        
        [
            isinstance(m, gpflow.models.ExternalDataTrainingLossMixin)
            for m in self.models
        ]
        objectives = [m.training_loss(d) for m, d in zip_longest(self.models, data)]
        return tf.reduce_sum(objectives)
    

    def training_loss(self, Xl, Yl) -> tf.function:
        """ Return a function """

        objectives=[]

        def closure() -> tf.Tensor:

            for i in range(len(self.models)):
                external = isinstance(self.models[i], gpflow.models.ExternalDataTrainingLossMixin)

                if external is False:
                    obj = self.models[i].training_loss()
                if external is True:
                    obj = self.models[i].training_loss((Xl[i],Yl[i]))
                objectives.append(obj)
        
            return tf.reduce_sum(objectives)

        if compile:
            closure = tf.function(closure)
        return  closure

class latent_Ensemble(latent_GPEnsemble):
    """
    Implements a range of Ensemble GP models.
    """

    def __init__(
        self,
        models: List[GPModel],
        method: EnsembleMethods,
        weighting: WeightingMethods,
        power: float = 8.0):
        """
        :param models: A list of GPflow models with the same prior and likelihood.
        """

        latent_GPEnsemble.__init__(self, models)
        self.method = method
        self.weighting = weighting
        self.power = power

    @check_shapes(
        "Xnew: [N, D]",
        "return[0]: [N, broadcast L]",
        "return[1]: [N, broadcast L]",
    )
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Prediction method based on the aggregation of multivariate submodel predictions.
        For more faster predictions and possibly more numerically stable predictions
        (but with lower accuracy), see the `predict_f_marginals` method.
        :param Xnew: 2D Array or tensor corresponding to points in the input
        where we want to make prediction.
        """
        assert not full_cov
        assert not full_output_cov
        b = "broadcast"

        # prior distribution
        
        #Xnew_ = Xnew * 100
        #_, var_p = self.models[0].predict_y(Xnew_)
        #vp = cs(var_p[:, None], f"[N, {b} L, {b} P]")
        
        vp_var = cs(self.models[0].kernel.kernels[0].K_diag(Xnew)[:, None, None], f"[N, {b} L, {b} P]") 
        vp_mu = cs(self.models[0].kernel.kernels[1].K_diag(Xnew)[:, None, None], f"[N, {b} L, {b} P]") 

        # expert distributions
        # P: number of experts
        preds = [m.predict_f(Xnew) for m in self.models]
        ## Mean function
        Me_mu = cs(
            tf.stack([pred[0][:, 0] for pred in preds], axis=1)[:, None, :], f"[N, {b} L, P]"
        )  # [n, latent, sub]
        Ve_mu = cs(tf.stack([pred[1][:, 0] for pred in preds], axis=1)[:, None, :], f"[N, {b} L, P]")
        
        ## Variance function
        Me_var = cs(
            tf.stack([pred[0][:, 1] for pred in preds], axis=1)[:, None, :], f"[N, {b} L, P]"
        )  # [n, latent, sub]
        Ve_var = cs(tf.stack([pred[1][:, 1] for pred in preds], axis=1)[:, None, :], f"[N, {b} L, P]")


        # Compute individual precisions - dim: n_experts x n_test_points
        prec_s_mu = cs(1.0 / Ve_mu, f"[N, {b} L, P]")
        prec_s_var = cs(1.0 / Ve_var, f"[N, {b} L, P]")

        weight_matrix_mu = cs(compute_weights(Me_mu, Ve_mu, self.power, self.weighting, vp_mu, softmax=True), f"[N, {b} L, P]")
        weight_matrix_var = cs(compute_weights(Me_var, Ve_var, self.power, self.weighting, vp_var, softmax=True), f"[N, {b} L, P]")

        # For all DgPs, normalized weights of experts requiring normalized weights and compute the aggegated local precisions
        if self.method == EnsembleMethods.POE:
            prec_mu = cs(tf.reduce_sum(prec_s_mu, axis=-1), f"[N, {b} L]")

        elif self.method == EnsembleMethods.GPOE:
            # weight_matrix = tf.linalg.normalize(weight_matrix, ord=1, axis=-1)
            weight_matrix = normalize_weights(weight_matrix)
            prec_mu = tf.reduce_sum(weight_matrix * prec_s_mu, axis=-1)

        elif self.method == EnsembleMethods.BCM:
            num_experts = tf.cast(tf.shape(vp_mu)[-1], vp_mu.dtype)
            prec_mu = tf.reduce_sum(prec_s_mu, axis=-1) + (1.0 - num_experts) / vp_mu[..., 0]
            prec_var = tf.reduce_sum(prec_s_var, axis=-1) + (1.0 - num_experts) / vp_var[..., 0]

        elif self.method == EnsembleMethods.RBCM:
            prec_mu = (
                tf.reduce_sum(weight_matrix_mu * prec_s_mu, axis=-1)
                + (1.0 - tf.reduce_sum(weight_matrix_mu, axis=-1)) / vp_mu[..., 0]
            )
            prec_var = (
                tf.reduce_sum(weight_matrix_var * prec_s_var, axis=-1)
                + (1.0 - tf.reduce_sum(weight_matrix_var, axis=-1)) / vp_var[..., 0]
            )

        # Compute the aggregated predictive means and variance of the barycenter
        if self.method == EnsembleMethods.BARY:
            # weight_matrix = tf.linalg.normalize(weight_matrix, ord=1, axis=-1)
            weight_matrix_mu = normalize_weights(weight_matrix_mu)
            weight_matrix_var = normalize_weights(weight_matrix_var)
            mu = tf.reduce_sum(weight_matrix_mu * Me_mu, axis=-1)
            var = tf.reduce_sum(weight_matrix_var * Ve_var, axis=-1)

        # For all DgPs compute the aggregated predictive means and variance
        else:
            prec_mu = cs(prec_mu, f"[N, {b} L]")
            unc_mu = 1.0 / prec_mu
            mu = unc_mu * tf.reduce_sum(weight_matrix_mu * prec_s_mu * Me_mu, axis=-1)

            prec_var = cs(prec_var, f"[N, {b} L]")
            unc_var = 1.0 / prec_var
            var = unc_var * tf.reduce_sum(weight_matrix_var * prec_s_var * Me_var, axis=-1)

        #np.save(str(self.weighting)+ "_weight_matrix_new.npy", weight_matrix.numpy())
            
        means = np.stack([mu, var], axis=1).squeeze()
        unc = np.stack([unc_mu, unc_var], axis=1).squeeze()

        #print(new_mu.shape, new_var.shape)

        return means, unc

    def predict_y(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        assert not full_cov 
        assert not full_output_cov

        m, unc = self.predict_f(Xnew)
        m = tf.constant(m, dtype=tf.float64) 
        unc = tf.constant(unc, dtype=tf.float64)
        return self.models[0].likelihood.predict_mean_and_var(Xnew, m, unc)

class OLD_FixedVarianceLikelihood(gpflow.functions.Function):
    """ Fixed Variance likelihood function"""
    def __init__(self, Y: gpflow.base.AnyNDArray):
        self.var_mean = np.var(Y, axis=-1, keepdims=True)

    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        return self.var_mean