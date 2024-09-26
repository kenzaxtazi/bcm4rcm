import numpy as np
import numpy as np
from scipy.stats import lognorm, gamma, expon, gaussian_kde
import matplotlib.pyplot as plt

__all__ = ['fit_lognormal', 'fit_kde',   'fit_gamma', 'fit_exponential', 'fit_distributions_to_4d_data', 'plot_fitted_distributions']

def fit_kde(data_points):
    """
    Fits a Kernel Density Estimate (KDE) to the data for a single location.

    Parameters
    ----------
    data_points : numpy.ndarray
        1D array of data points for the specified location.

    Returns
    -------
    kde : gaussian_kde
        The fitted KDE object.

    log_likelihood : float
        The log-likelihood of the fitted KDE model.
    """
    # Fit the Kernel Density Estimate
    kde = gaussian_kde(data_points)

    # Evaluate the estimated density at the data points
    densities = kde(data_points)

    # Handle zero densities to avoid log(0)
    non_zero_densities = densities > 0
    if not np.all(non_zero_densities):
        print("Warning: Zero density encountered in KDE. Ignoring zero densities in log-likelihood.")
    
    log_likelihood = np.sum(np.log(densities[non_zero_densities]))

    return kde, log_likelihood

def fit_lognormal(data):
    """
    Fits a log-normal distribution to the data.

    Parameters
    ----------
    data : numpy.ndarray
        1D array of positive data points.

    Returns
    -------
    shape : float
        The shape parameter of the fitted log-normal distribution.

    loc : float
        The location parameter of the fitted log-normal distribution.

    scale : float
        The scale parameter of the fitted log-normal distribution.

    log_likelihood : float
        The log-likelihood of the fitted model.
    """
    # Check for positive data
    if np.any(data <= 0):
        raise ValueError("Data must be positive for log-normal distribution.")

    shape, loc, scale = lognorm.fit(data, floc=0)
    log_likelihood = np.sum(lognorm.logpdf(data, shape, loc=loc, scale=scale))
    return shape, loc, scale, log_likelihood

def fit_gamma(data):
    """
    Fits a gamma distribution to the data.

    Parameters
    ----------
    data : numpy.ndarray
        1D array of positive data points.

    Returns
    -------
    shape : float
        The shape parameter of the fitted gamma distribution.

    loc : float
        The location parameter (should be zero if fixed).

    scale : float
        The scale parameter of the fitted gamma distribution.

    log_likelihood : float
        The log-likelihood of the fitted model.
    """
    # Check for positive data
    if np.any(data <= 0):
        raise ValueError("Data must be positive for gamma distribution.")

    shape, loc, scale = gamma.fit(data, floc=0)
    log_likelihood = np.sum(gamma.logpdf(data, shape, loc=loc, scale=scale))
    return shape, loc, scale, log_likelihood

def fit_exponential(data):
    """
    Fits an exponential distribution to the data.

    Parameters
    ----------
    data : numpy.ndarray
        1D array of data points.

    Returns
    -------
    loc : float
        The location parameter (should be zero if fixed).

    scale : float
        The scale parameter of the fitted exponential distribution.

    log_likelihood : float
        The log-likelihood of the fitted model.
    """
    # Check for data greater than or equal to zero
    if np.any(data < 0):
        raise ValueError("Data must be non-negative for exponential distribution.")

    loc, scale = expon.fit(data, floc=0)
    log_likelihood = np.sum(expon.logpdf(data, loc=loc, scale=scale))
    return loc, scale, log_likelihood

def fit_distributions_to_4d_data(data, distributions=None):
    """
    Fits specified distributions to each location in a 4D data array.

    Parameters
    ----------
    data : numpy.ndarray
        A 4-dimensional NumPy array where the last dimension contains the data points.

    distributions : list of str, optional
        A list of distribution names to fit. Options are 'lognormal', 'gamma',
        'exponential', and 'kde'. If None, all distributions are fitted.

    Returns
    -------
    results : dict
        A dictionary containing the fitted parameters and log-likelihoods for each
        specified distribution and KDE.
    """
    import numpy as np

    # Default to all distributions if none specified
    if distributions is None:
        distributions = ['lognormal', 'gamma', 'exponential', 'kde']

    # Validate input distributions
    valid_distributions = {'lognormal', 'gamma', 'exponential', 'kde'}
    invalid_distributions = set(distributions) - valid_distributions
    if invalid_distributions:
        raise ValueError(f"Invalid distributions specified: {invalid_distributions}")

    # Get the shape of the input data
    dims = data.shape
    if len(dims) != 4:
        raise ValueError("Input data must be a 4-dimensional array.")

    # Initialize a dictionary to store the results
    results = {}

    if 'lognormal' in distributions:
        results['lognormal'] = {
            'shape': np.zeros(dims[:-1]),
            'loc': np.zeros(dims[:-1]),
            'scale': np.zeros(dims[:-1]),
            'log_likelihood': np.zeros(dims[:-1]),
        }
    if 'gamma' in distributions:
        results['gamma'] = {
            'shape': np.zeros(dims[:-1]),
            'loc': np.zeros(dims[:-1]),
            'scale': np.zeros(dims[:-1]),
            'log_likelihood': np.zeros(dims[:-1]),
        }
    if 'exponential' in distributions:
        results['exponential'] = {
            'loc': np.zeros(dims[:-1]),
            'scale': np.zeros(dims[:-1]),
            'log_likelihood': np.zeros(dims[:-1]),
        }
    if 'kde' in distributions:
        results['kde'] = {
            'kde_objects': {},  # Store KDE objects with location keys
            'log_likelihood': np.zeros(dims[:-1]),
        }

    # Iterate through each location
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                # Extract the data points for the current location
                data_points = data[i, j, k, :]

                # Handle potential invalid data
                if np.any(np.isnan(data_points)) or len(data_points) == 0:
                    print(f"Warning: Invalid data at location ({i}, {j}, {k}). Skipping.")
                    continue

                # Fit the specified distributions
                if 'lognormal' in distributions:
                    try:
                        shape_ln, loc_ln, scale_ln, ll_ln = fit_lognormal(data_points)
                        results['lognormal']['shape'][i, j, k] = shape_ln
                        results['lognormal']['loc'][i, j, k] = loc_ln
                        results['lognormal']['scale'][i, j, k] = scale_ln
                        results['lognormal']['log_likelihood'][i, j, k] = ll_ln
                    except Exception as e:
                        print(f"Log-normal fitting failed at location ({i}, {j}, {k}): {e}")

                if 'gamma' in distributions:
                    try:
                        shape_g, loc_g, scale_g, ll_g = fit_gamma(data_points)
                        results['gamma']['shape'][i, j, k] = shape_g
                        results['gamma']['loc'][i, j, k] = loc_g
                        results['gamma']['scale'][i, j, k] = scale_g
                        results['gamma']['log_likelihood'][i, j, k] = ll_g
                    except Exception as e:
                        print(f"Gamma fitting failed at location ({i}, {j}, {k}): {e}")

                if 'exponential' in distributions:
                    try:
                        loc_exp, scale_exp, ll_exp = fit_exponential(data_points)
                        results['exponential']['loc'][i, j, k] = loc_exp
                        results['exponential']['scale'][i, j, k] = scale_exp
                        results['exponential']['log_likelihood'][i, j, k] = ll_exp
                    except Exception as e:
                        print(f"Exponential fitting failed at location ({i}, {j}, {k}): {e}")

                if 'kde' in distributions:
                    try:
                        kde, ll_kde = fit_kde(data_points)
                        results['kde']['kde_objects'][(i, j, k)] = kde
                        results['kde']['log_likelihood'][i, j, k] = ll_kde
                    except Exception as e:
                        print(f"KDE fitting failed at location ({i}, {j}, {k}): {e}")

    return results

def plot_fitted_distributions(data, fitted_results, location, distributions=None, ax=None):
    """
    Plots the histogram of data points and the specified fitted distributions 
    for a given spatiotemporal location. If an Axes object is provided, it plots on that Axes;
    otherwise, it creates a new figure and plots there.

    Parameters
    ----------
    data : numpy.ndarray
        A 4-dimensional NumPy array where the last dimension contains the data points.

    fitted_results : dict
        A dictionary containing the fitted parameters and distributions for each distribution,
        as returned by the fit_distributions_to_4d_data function.

    location : tuple
        A tuple (i, j, k) specifying the spatiotemporal location to plot.

    distributions : list of str, optional
        A list of distribution names to plot. Options are 'lognormal', 'gamma',
        'exponential', and 'kde'. If None, all available distributions in fitted_results are plotted.

    ax : matplotlib.axes.Axes, optional
        An Axes object on which to plot. If None, a new figure and axes are created.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import lognorm, gamma, expon

    i, j, k = location
    data_points = data[i, j, k, :]

    # Default to all distributions present in fitted_results if none specified
    if distributions is None:
        distributions = list(fitted_results.keys())

    # Validate input distributions
    valid_distributions = {'lognormal', 'gamma', 'exponential', 'kde'}
    invalid_distributions = set(distributions) - valid_distributions
    if invalid_distributions:
        raise ValueError(f"Invalid distributions specified for plotting: {invalid_distributions}")

    # Create an array for x values for plotting
    x_min = np.min(data_points)
    x_max = np.max(data_points)
    x = np.linspace(x_min, x_max, 1000)

    # If ax is None, create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Plotting
    # Histogram of data points
    ax.hist(data_points, bins=30, density=True, alpha=0.5, color='gray', label='Data Histogram')

    # Plot the specified distributions
    plotted = False  # Flag to check if any distributions are plotted

    if 'lognormal' in distributions and 'lognormal' in fitted_results:
        # Extract fitted parameters
        shape_ln = fitted_results['lognormal']['shape'][i, j, k]
        loc_ln = fitted_results['lognormal']['loc'][i, j, k]
        scale_ln = fitted_results['lognormal']['scale'][i, j, k]
        # Evaluate PDF
        pdf_lognormal = lognorm.pdf(x, s=shape_ln, loc=loc_ln, scale=scale_ln)
        ax.plot(x, pdf_lognormal, 'r-', lw=2, label='Fitted Log-Normal')
        plotted = True

    if 'gamma' in distributions and 'gamma' in fitted_results:
        # Extract fitted parameters
        shape_g = fitted_results['gamma']['shape'][i, j, k]
        loc_g = fitted_results['gamma']['loc'][i, j, k]
        scale_g = fitted_results['gamma']['scale'][i, j, k]
        # Evaluate PDF
        pdf_gamma = gamma.pdf(x, a=shape_g, loc=loc_g, scale=scale_g)
        ax.plot(x, pdf_gamma, 'g-', lw=2, label='Fitted Gamma')
        plotted = True

    if 'exponential' in distributions and 'exponential' in fitted_results:
        # Extract fitted parameters
        loc_exp = fitted_results['exponential']['loc'][i, j, k]
        scale_exp = fitted_results['exponential']['scale'][i, j, k]
        # Evaluate PDF
        pdf_exponential = expon.pdf(x, loc=loc_exp, scale=scale_exp)
        ax.plot(x, pdf_exponential, 'b-', lw=2, label='Fitted Exponential')
        plotted = True

    if 'kde' in distributions and 'kde' in fitted_results:
        # Get KDE object
        kde = fitted_results['kde']['kde_objects'].get((i, j, k), None)
        if kde is not None:
            # Evaluate KDE
            pdf_kde = kde(x)
            ax.plot(x, pdf_kde, 'k--', lw=2, label='Kernel Density Estimate')
            plotted = True
        else:
            print(f"No KDE available for location ({i}, {j}, {k}).")

    if not plotted:
        print(f"No distributions to plot for location ({i}, {j}, {k}).")
        return

    # Add titles and labels
    ax.set_title(f'Fitted Distributions at Location ({i}, {j}, {k})')
    ax.set_xlabel('Data Values')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)

    # If a new figure was created (ax was None), show the plot
    if ax is None:
        plt.show()

if __name__ == "__main__":
    # Example usage
    # Generate some random data for testing
    np.random.seed(0)
    data_array = np.random.lognormal(mean=1.0, sigma=0.5, size=(12, 90, 40, 3000))

    # Fit the distributions to the data
    fit_results = fit_distributions_to_4d_data(data_array)

    # Verify the results
    print("Log-normal Parameters:\n", fit_results['lognormal'])
    print("Gamma Parameters:\n", fit_results['gamma'])
    print("Exponential Parameters:\n", fit_results['exponential'])