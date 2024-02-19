import numpy as np



def wass_distance(P_mu, P_var, Q_mu, Q_var):
    """ The 2-Wassertein distance between two normal distributions """
    wass =  (P_mu - Q_mu)**2 + (P_var - Q_var)**2
    return wass


def rcm_wass_distances(rcm_arr: np.ndarray, aphro_arr:np.ndarray, save:bool=False)-> np.ndarray:
    """
    Returns 2-Wasserstein distances between RCM and APHRODITE data.

    Args:
        rcm_list (list): scaled RCM data
        aphrodite (np.ndarray): scaled APHRODITE data

    Returns:
        np.ndarray: Wasserstein distances (not normalised) 
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

