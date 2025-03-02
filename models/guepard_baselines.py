"""
Code based on guepard
https://github.com/NicolasDurrande/guepard/tree/main/guepard
guepard/baselines.py

Additional chained GP implementation (LatentEnsemble and LatentGPEnsemble classes).

"""

import abc
import gpflow
import tensorflow as tf
import cs.check_shapes as cs

from enum import Enum
from itertools import zip_longest
from gpflow.models import GPModel
from typing import List, Optional, Union
from gpflow.base import InputData, MeanAndVariance, RegressionData


class EnsembleMethods(Enum):
    """Aggregation methods"""

    POE = "PoE"
    GPOE = "gPoE"
    BCM = "BCM"
    RBCM = "rBCM"
    BARY = "Bary"
    MOE = "MOE"


class WeightingMethods(Enum):
    """Weighting methods"""

    VAR = "Var"
    WASS = "Wasser"
    UNI = "Uniform"
    ENT = "Entropy"
    NONE = "NoWeights"
    KL = "KL"


@cs(
    "mu_s: [N, L, P]  # N: num data, L: num latent, P: num experts",
    "var_s: [N, L, P]",
    "power: []",
    "prior_var: [N, broadcast L, broadcast P]",
    "return: [N, L, P]",
)
def compute_weights(
    mu_s: tf.Tensor,
    var_s: tf.Tensor,
    power: float,
    weighting: WeightingMethods,
    prior_var: Optional[tf.Tensor] = None,
    softmax: bool = False,
) -> tf.Tensor:
    """
    Compute unnormalized weight matrix.

    Args:
        mu_s (tf.Tensor): predictive mean of each expert (P) at each test point (N) for each
            output (L)
        var_s (tf.Tensor): predictive (marginal) variance of each expert (P) at each test point (N)
            for each output (L)
        power (float): scalar, softmax scaling
        weighting (WeightingMethods): weighting method
        prior_var (Optional[tf.Tensor], optional): prior variance. Defaults to None.
        softmax (bool, optional): whether to use softmax scaling or fraction scaling.
            Defaults to False.

    Raises:
        NotImplementedError: Unknown weighting passed to compute_weights.

    Returns:
        tf.Tensor: weight_matrix, dimension: n_expert x n_test_points : unnormalized weight of
          ith expert at jth test point
    """

    if weighting == WeightingMethods.VAR:
        return tf.math.exp(-power * var_s)

    elif weighting == WeightingMethods.WASS:
        wass = mu_s**2 + (var_s - prior_var)  # ** 2
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


def normalize_weights(weight_matrix: tf.Tensor) -> tf.Tensor:
    """Normalize weight matrix."""
    sum_weights = tf.reduce_sum(weight_matrix, axis=-1, keepdims=True)
    weight_matrix = weight_matrix / sum_weights
    return weight_matrix


class GPEnsemble(GPModel, metaclass=abc.ABCMeta):
    """Base class for GP ensembles."""

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
        """Return trainable variables."""
        r = []
        for model in self.models:
            r += model.trainable_variables
        return r

    def maximum_log_likelihood_objective(
        self, data: List[RegressionData]
    ) -> tf.Tensor:  # type: ignore
        """Return maximum log likelihood objective from data."""
        _ = [
            isinstance(m, gpflow.models.ExternalDataTrainingLossMixin)
            for m in self.models
        ]
        objectives = [m.training_loss(d) for m, d in zip_longest(self.models, data)]
        return tf.reduce_sum(objectives)

    def training_loss(
        self, data: List[Union[None, RegressionData]] = [None]
    ) -> tf.Tensor:
        """Return training loss function."""
        external = [
            isinstance(m, gpflow.models.ExternalDataTrainingLossMixin)
            for m in self.models
        ]
        objectives = [
            m.training_loss(d) if ext else m.training_loss()
            for m, ext, d in zip_longest(self.models, external, data)
        ]
        return tf.reduce_sum(objectives)


class LatentGPEnsemble(GPModel, metaclass=abc.ABCMeta):
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
        """Return trainable variables."""
        r = []
        for model in self.models:
            r += model.trainable_variables
        return r

    @property
    def variational_variables(self):  # type: ignore
        """Return variational variables."""
        r = []
        for model in self.models:
            r.append([model.q_mu, model.q_sqrt])
        return r

    def maximum_log_likelihood_objective(
        self, data: List[RegressionData]
    ) -> tf.Tensor:  # type: ignore
        """This give a scalar value, data is a zipped list of X and Y"""
        _ = [
            isinstance(m, gpflow.models.ExternalDataTrainingLossMixin)
            for m in self.models
        ]
        objectives = [m.training_loss(d) for m, d in zip_longest(self.models, data)]
        return tf.reduce_sum(objectives)

    def training_loss(self, Xl, Yl) -> tf.function:
        """Return a training loss function."""
        objectives = []

        def closure() -> tf.Tensor:
            for i in enumerate(self.models):
                external = isinstance(
                    self.models[i], gpflow.models.ExternalDataTrainingLossMixin
                )

                if external is False:
                    obj = self.models[i].training_loss()
                if external is True:
                    obj = self.models[i].training_loss((Xl[i], Yl[i]))
                objectives.append(obj)
            return tf.reduce_sum(objectives)

        if compile:
            closure = tf.function(closure)
        return closure


class Ensemble(GPEnsemble):
    """
    Implements a range of Ensemble GP models.
    """

    def __init__(
        self,
        models: List[GPModel],
        method: EnsembleMethods,
        weighting: WeightingMethods,
        power: float = 8.0,
    ):
        """
        :param models: A list of GPflow models with the same prior and likelihood.
        """

        GPEnsemble.__init__(self, models)
        self.method = method
        self.weighting = weighting
        self.power = power

    @cs(
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
        vp = cs(self.models[0].kernel.K_diag(Xnew)[:, None, None], f"[N, {b} L, {b} P]")

        # expert distributions
        # P: number of experts
        preds = [m.predict_f(Xnew) for m in self.models]
        Me = cs(
            tf.stack([pred[0] for pred in preds], axis=2), f"[N, {b} L, P]"
        )  # [n, latent, sub]
        Ve = cs(tf.stack([pred[1] for pred in preds], axis=2), f"[N, {b} L, P]")

        # Compute individual precisions - dim: n_experts x n_test_points
        prec_s = cs(1.0 / Ve, f"[N, {b} L, P]")

        weight_matrix = cs(
            compute_weights(Me, Ve, self.power, self.weighting, vp, softmax=True),
            f"[N, {b} L, P]",
        )

        # For all DgPs, normalized weights of experts requiring normalized weights
        # and compute the aggegated local precisions
        if self.method == EnsembleMethods.POE:
            prec = cs(tf.reduce_sum(prec_s, axis=-1), f"[N, {b} L]")

        elif self.method == EnsembleMethods.GPOE:
            # weight_matrix = tf.linalg.normalize(weight_matrix, ord=1, axis=-1)
            weight_matrix = normalize_weights(weight_matrix)
            prec = tf.reduce_sum(weight_matrix * prec_s, axis=-1)

        elif self.method == EnsembleMethods.BCM:
            num_experts = tf.cast(tf.shape(vp)[-1], vp.dtype)
            prec = tf.reduce_sum(prec_s, axis=-1) + (1.0 - num_experts) / vp[..., 0]

        elif self.method == EnsembleMethods.RBCM:
            prec = (
                tf.reduce_sum(weight_matrix * prec_s, axis=-1)
                + (1.0 - tf.reduce_sum(weight_matrix, axis=-1)) / vp[..., 0]
            )

        # Compute the aggregated predictive means and variance of the barycenter
        if self.method == EnsembleMethods.BARY:
            # weight_matrix = tf.linalg.normalize(weight_matrix, ord=1, axis=-1)
            weight_matrix = normalize_weights(weight_matrix)
            mu = tf.reduce_sum(weight_matrix * Me, axis=-1)
            var = tf.reduce_sum(weight_matrix * Ve, axis=-1)

        # For all DgPs compute the aggregated predictive means and variance
        else:
            prec = cs(prec, f"[N, {b} L]")
            var = 1.0 / prec
            mu = var * tf.reduce_sum(weight_matrix * prec_s * Me, axis=-1)

        # np.save(str(self.weighting)+ "_weight_matrix_new.npy", weight_matrix.numpy())

        return mu, var

    def predict_y(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """Predict mean and variance given input Xnew."""
        assert not full_cov
        assert not full_output_cov

        m, v = self.predict_f(Xnew)
        return self.likelihood.predict_mean_and_var(Xnew, m, v)


class LatentEnsemble(LatentGPEnsemble):
    """
    Implements a range of Ensemble GP models.
    """

    def __init__(
        self,
        models: List[GPModel],
        method: EnsembleMethods,
        weighting: WeightingMethods,
        power: float = 8.0,
    ):
        """
        :param models: A list of GPflow models with the same prior and likelihood.
        """

        LatentGPEnsemble.__init__(self, models)
        self.method = method
        self.weighting = weighting
        self.power = power

    @cs(
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
        vp = cs(self.models[0].kernel.K_diag(Xnew)[:, None, None], f"[N, {b} L, {b} P]")

        # expert distributions
        # P: number of experts
        preds = [m.predict_f(Xnew) for m in self.models]
        Me = cs(
            tf.stack([pred[0] for pred in preds], axis=2), f"[N, {b} L, P]"
        )  # [n, latent, sub]
        Ve = cs(tf.stack([pred[1] for pred in preds], axis=2), f"[N, {b} L, P]")

        # Compute individual precisions - dim: n_experts x n_test_points
        prec_s = cs(1.0 / Ve, f"[N, {b} L, P]")

        weight_matrix = cs(
            compute_weights(Me, Ve, self.power, self.weighting, vp, softmax=True),
            f"[N, {b} L, P]",
        )

        prec = None

        # For all DgPs, normalized weights of experts requiring normalized weights
        # and compute the aggegated local precisions
        if self.method == EnsembleMethods.POE:
            prec = cs(tf.reduce_sum(prec_s, axis=-1), f"[N, {b} L]")

        elif self.method == EnsembleMethods.GPOE:
            # weight_matrix = tf.linalg.normalize(weight_matrix, ord=1, axis=-1)
            weight_matrix = normalize_weights(weight_matrix)
            prec = tf.reduce_sum(weight_matrix * prec_s, axis=-1)

        elif self.method == EnsembleMethods.BCM:
            num_experts = tf.cast(tf.shape(vp)[-1], vp.dtype)
            prec = tf.reduce_sum(prec_s, axis=-1) + (1.0 - num_experts) / vp[..., 0]

        elif self.method == EnsembleMethods.RBCM:
            prec = (
                tf.reduce_sum(weight_matrix * prec_s, axis=-1)
                + (1.0 - tf.reduce_sum(weight_matrix, axis=-1)) / vp[..., 0]
            )

        # Compute the aggregated predictive means and variance of the barycenter
        if self.method == EnsembleMethods.BARY:
            # weight_matrix = tf.linalg.normalize(weight_matrix, ord=1, axis=-1)
            weight_matrix = normalize_weights(weight_matrix)
            mu = tf.reduce_sum(weight_matrix * Me, axis=-1)
            var = tf.reduce_sum(weight_matrix * Ve, axis=-1)

        # For all DgPs compute the aggregated predictive means and variance
        else:
            prec = cs(prec, f"[N, {b} L]")
            var = 1.0 / prec
            mu = var * tf.reduce_sum(weight_matrix * prec_s * Me, axis=-1)

        # np.save(str(self.weighting)+ "_weight_matrix_new.npy", weight_matrix.numpy())

        return mu, var

    def predict_y(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """Predict mean and variance for the given input Xnew."""
        assert not full_cov
        assert not full_output_cov

        m, v = self.predict_f(Xnew)
        return self.likelihood.predict_mean_and_var(Xnew, m, v)


def get_gpr_submodels(
    data_list,
    kernel,
    mean_function=None,
    noise_variance: float = 0.01,
) -> List:
    """
    Helper function to build a list of GPflow GPR submodels from a list of datasets,
     a GP prior and a likelihood variance.
    """
    models = [
        gpflow.models.GPR(data, kernel, mean_function, noise_variance)
        for data in data_list
    ]
    # else:
    # models = [GPR((np.mean(data[0], axis=-2),
    #   np.mean(data[1], axis=-2)),
    #   kernel, mean_function,
    #   likelihood=gpflow.likelihoods.Gaussian(FixedVarianceOfMean(data[1]))) for data in data_list]
    for m in models[1:]:
        m.likelihood = models[0].likelihood
        m.mean_function = models[0].mean_function
    return models
