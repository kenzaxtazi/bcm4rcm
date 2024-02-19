import numpy as np


def wass_distance(P_mu:np.float64, P_var:np.float64, Q_mu:np.float64, Q_var:np.float64)-> np.float64:
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
    The data should have already been scaled by it 95th percentile value
    followed by a Box-Cox transformation.

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
        wass = wass_distance(P_mu, P_var, Q_mu, Q_var)
        wass_dists.append(wass)
    
    wass_arr = np.array(wass_dists)

    if save:
        np.save('wass_dists.npy', wass_arr)
    
    return wass_arr


def softmax(wass:np.ndarray, T:int=8)-> np.array:
    """
    Generate model weights. The higher the temperature T the more 
    likely RCMs are to each other.

    Args:
        wass (np.array): Wasserstain-2 distances (RCM x months x lat x lon x 1).
        T (int, optional): temperature. Defaults to 8. 

    Returns:
        np.array: model weights (RCM x months x lat x lon x 1)
    """
    weights = np.exp(wass / T)
    # take weights for each model and normalise them by the sum of all weights
    weight_sum = np.sum(weights, axis=0)
    weights_norm = weights / weight_sum
    return weights_norm


def mixture_of_experts(weights:np.array, model_outputs:np.array)-> np.array:
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