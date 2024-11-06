import numpy as np
import scipy as sp


def normal_wass_distance(P_mu:np.float64, P_var:np.float64, Q_mu:np.float64, Q_var:np.float64)-> np.float64:
    """
    The 2-Wassertein distance between two normal distributions.

    Args:
        P_mu (np.float64): mean of distribution P.
        P_var (np.float64): variance of distribution P.
        Q_mu (np.float64): mean of distribution Q.
        Q_var (np.float64): variance of distribution Q.

    Returns:
        np.float64: 2-Wasserstein distance.
    """
    wass =  (P_mu - Q_mu)**2 + (P_var - Q_var)**2
    return wass


def rcm_wass_distances(rcm_arr: np.ndarray, aphro_arr:np.ndarray, save:bool=False)-> np.ndarray:
    """
    Returns 2-Wasserstein distances between RCM and APHRODITE data. 
    The data should have already been scaled by it 95th percentile value.

    Args:
        rcm_list (np.ndarray): scaled RCM data. Shape RCM * years * months * lat * lon (5, 30, 12, 90, 40).
        aphrodite (np.ndarray): scaled APHRODITE data.

    Returns:
        np.ndarray: Wasserstein distances (not normalised). 
    """

    wass_dists = []

    # calculate wass dist using scipy
    # TODO: vectorise this

    for i in range(rcm_arr.shape[0]):
        for m in range(rcm_arr.shape[2]):
            for lon in range(rcm_arr.shape[3]):
                for lat in range(rcm_arr.shape[4]):
                    wass = sp.stats.wasserstein_distance(rcm_arr[i,:, m, lon, lat], aphro_arr[:, m, lon, lat])
                    wass_dists.append(wass)
                    
    wass_arr = np.array(wass_dists)

    if save:
        np.save('wass_dists.npy', wass_arr)
    
    return wass_arr

def rcm_normal_wass_distances(rcm_arr: np.ndarray, aphro_arr:np.ndarray, save:bool=False)-> np.ndarray:
    """
    Returns 2-Wasserstein distances between RCM and APHRODITE data. Assuming both distributions are normal.
    The data should have already been scaled by it 95th percentile value.

    Args:
        rcm_list (np.ndarray): scaled RCM data.
        aphrodite (np.ndarray): scaled APHRODITE data.

    Returns:
        np.ndarray: Wasserstein distances (not normalised). 
    """

    wass_dists = []

    for i in range(len(rcm_arr)):
        P_var = np.var(rcm_arr[i], axis=-2)
        P_mu = np.mean(rcm_arr[i], axis=-2)
        Q_var = np.var(aphro_arr, axis=-2)
        Q_mu = np.mean(aphro_arr, axis=-2)
        wass = normal_wass_distance(P_mu, P_var, Q_mu, Q_var)
        wass_dists.append(wass)
    
    wass_arr = np.array(wass_dists)

    if save:
        np.save('wass_dists.npy', wass_arr)
    
    return wass_arr


def softmax_weighting(wass:np.ndarray, T:int=1)-> np.array:
    """
    Generate model weights. The higher the temperature T the more 
    likely RCMs are to each other.

    Args:
        wass (np.array): Wasserstain distances (RCM x months x lat x lon x 1).
        T (int, optional): temperature. Defaults to 8. 

    Returns:
        np.array: model weights (RCM x months x lat x lon x 1)
    """
    weights = np.exp(-wass / T)
    # take weights for each model and normalise them by the sum of all weights
    weight_sum = np.sum(weights, axis=0)
    weights_norm = weights / weight_sum
    return weights_norm

def inverse_weighting(wass:np.ndarray)-> np.array:
    """
    Generate model weights. The higher the temperature T the more 
    likely RCMs are to each other.

    Args:
        wass (np.array): Wasserstain distances (RCM x months x lat x lon x 1).
    Returns:
        np.array: model weights (RCM x months x lat x lon x 1)
    """
    weights = 1 / wass
    # take weights for each model and normalise them by the sum of all weights
    weight_sum = np.sum(weights, axis=0)
    weights_norm = weights / weight_sum
    return weights_norm


def OLD_mean_mixture_of_experts(weights:np.array, model_outputs:np.ndarray)-> np.ndarray:
    """
    Generate mixture of experts.

    Args:
        weights (np.array): weights for all models.
        model_outputs (np.array): model outputs.

    Returns:
        np.array: mixture of experts mean
    """
    mean = np.sum(weights * model_outputs, axis=0)
    return mean


class mixture_of_experts():
    """ Generate mixture of experts model for arbitrary distributions. """

    def __init__(self, lambdas:np.ndarray, means:np.ndarray, vars:np.ndarray, weights:np.ndarray):
        """
        Intialise the mixture of experts model for RCMs.

        Args:
            lambdas (np.ndarray): Box-Cox scaling factors for each of the 5 distributions.
            means (np.ndarray): Means of the normals for the BCM4RCMs.
            vars (np.ndarray): Variances of the normals for the BCM4RCMs.
            weights (np.array): Weights assigned to each of the BCM4RCMs.
        """
        self.lambdas = lambdas
        self.means = means
        self.vars = vars
        self.weights = weights


    def predict_prob(self, x:np.ndarray)-> np.ndarray:
        """
        Predict the probability density function of the mixture of experts model.

        Args:
            x (np.ndarray): precipitation values.

        Returns:
            np.ndarray: predicted probabilities.
        """
        rcm_probs = []

        n = len(self.lambdas)

        for i in range(n):
            normal_dists = sp.stats.norm(self.means[i], self.vars[i]).pdf(x)
            probs = sp.stats.spcecial.inv_boxcox(normal_dists, self.lambdas[i])
            rcm_probs.append(probs)

        rcm_probs = np.array(rcm_probs)
        moe_probs = np.sum(self.weights * rcm_probs, axis=0)

        return moe_probs