import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import anderson_ksamp

"""Set of functions to compute eval metrics between corresponding empirical distributions in x and y arrays."""


def compute_kolmogorov_smirnov(x, y):
    """
    Compute the Kolmogorov-Smirnov test between corresponding distributions in x and y.

    Parameters:
    x, y: numpy arrays of shape (..., N)
        Arrays containing the samples of the empirical distributions.
        The last dimension corresponds to the samples.

    Returns:
    ks_statistic: numpy array of shape (...)
        The Kolmogorov-Smirnov test statistic at each grid point.
    p_value: numpy array of shape (...)
        The p-value of the test at each grid point.
    """
    # Ensure the input arrays have the same shape
    if x.shape != y.shape:
        raise ValueError("Input arrays x and y must have the same shape.")

    # Get the shape of the grid (excluding the samples dimension)
    grid_shape = x.shape[:-1]
    N = x.shape[-1]  # Number of samples in each distribution

    # Flatten the grid dimensions to iterate over distributions
    x_flat = x.reshape(-1, N)
    y_flat = y.reshape(-1, N)
    num_distributions = x_flat.shape[0]

    # Initialize arrays to store the test statistics and p-values
    ks_statistic = np.zeros(num_distributions)
    p_value = np.zeros(num_distributions)

    # Loop over each pair of corresponding distributions
    for i in range(num_distributions):
        # Perform the Kolmogorov-Smirnov test
        result = ks_2samp(x_flat[i], y_flat[i])

        # Store the test statistic and p-value
        ks_statistic[i] = result.statistic
        p_value[i] = result.pvalue

    # Reshape the results back to the grid shape
    ks_statistic = ks_statistic.reshape(grid_shape)
    p_value = p_value.reshape(grid_shape)

    return ks_statistic, p_value


def compute_anderson_darling(x, y):
    """
    Compute the Anderson-Darling test between corresponding distributions in x and y.

    Parameters:
    x, y: numpy arrays of shape (..., N)
        Arrays containing the samples of the empirical distributions.
        The last dimension corresponds to the samples.

    Returns:
    ad_statistic: numpy array of shape (...)
        The Anderson-Darling test statistic at each grid point.
    p_value: numpy array of shape (...)
        The p-value of the test at each grid point.
    """
    # Ensure the input arrays have the same shape
    if x.shape != y.shape:
        raise ValueError("Input arrays x and y must have the same shape.")

    # Get the shape of the grid (excluding the samples dimension)
    grid_shape = x.shape[:-1]
    N = x.shape[-1]  # Number of samples in each distribution

    # Flatten the grid dimensions to iterate over distributions
    x_flat = x.reshape(-1, N)
    y_flat = y.reshape(-1, N)
    num_distributions = x_flat.shape[0]

    # Initialize arrays to store the test statistics and p-values
    ad_statistic = np.zeros(num_distributions)
    p_value = np.zeros(num_distributions)

    # Loop over each pair of corresponding distributions
    for i in range(num_distributions):
        # Perform the Anderson-Darling test
        result = anderson_ksamp([x_flat[i], y_flat[i]])

        # Store the test statistic and p-value
        ad_statistic[i] = result.statistic
        p_value[i] = (
            result.significance_level / 100.0
        )  # Convert percentage to proportion

    # Reshape the results back to the grid shape
    ad_statistic = ad_statistic.reshape(grid_shape)
    p_value = p_value.reshape(grid_shape)

    return ad_statistic, p_value


def compute_crps(x, y):
    """
    Compute element-wise CRPS between corresponding empirical distributions in x and y.

    Parameters:
    x, y: numpy arrays of shape (..., N_x) and (..., N_y)
        Arrays containing the samples of the empirical distributions.
        The last dimension corresponds to the samples.

    Returns:
    crps: numpy array of shape (...)
        The CRPS between the empirical distributions at each grid point.
    """
    # Determine the number of samples for x and y
    N_x = x.shape[-1]
    N_y = y.shape[-1]

    # Compute term1: Mean absolute difference between all pairs of x and y
    # Shape after broadcasting: (..., N_x, N_y)
    diff_xy = np.abs(x[..., :, np.newaxis] - y[..., np.newaxis, :])
    term1 = np.mean(diff_xy, axis=(-2, -1))

    # Compute term2: 0.5 * Mean absolute difference within x
    if N_x > 1:
        diff_xx = np.abs(
            x[..., :, np.newaxis] - x[..., np.newaxis, :]
        )  # Shape: (..., N_x, N_x)
        # To avoid double-counting and self-differences, subtract the diagonal
        sum_diff_xx = (
            np.sum(diff_xx, axis=(-2, -1)) - N_x * 0
        )  # Diagonal is zero since |x_i - x_i| = 0
        # Number of unique pairs: N_x * (N_x - 1)
        term2 = 0.5 * (sum_diff_xx / (N_x * N_x))
    else:
        term2 = 0.0  # If there's only one sample, intra-difference is zero

    # Compute term3: 0.5 * Mean absolute difference within y
    if N_y > 1:
        diff_yy = np.abs(
            y[..., :, np.newaxis] - y[..., np.newaxis, :]
        )  # Shape: (..., N_y, N_y)
        sum_diff_yy = np.sum(diff_yy, axis=(-2, -1)) - N_y * 0  # Diagonal is zero
        term3 = 0.5 * (sum_diff_yy / (N_y * N_y))
    else:
        term3 = 0.0  # If there's only one sample, intra-difference is zero

    # Compute CRPS
    crps = term1 - term2 - term3
    return crps


def compute_brier_score(x, y, threshold):
    """
    Compute the Brier score between corresponding distributions in x and y,
    given an event threshold.

    Parameters:
    x, y: numpy arrays of shape (..., N)
        Arrays containing the samples of the empirical distributions.
        The last dimension corresponds to the samples.
    threshold: float
        The event threshold to define the occurrence of the event.

    Returns:
    brier_score: numpy array of shape (...)
        The Brier score at each grid point.
    """
    # Compute forecast probabilities p_f
    p_f = np.mean(x > threshold, axis=-1)

    # Compute observed probabilities p_o
    p_o = np.mean(y > threshold, axis=-1)

    # Compute Brier score
    brier_score = (p_f - p_o) ** 2

    return brier_score
