import pandas as pd

def aggregate_to_4d_array(df, agg_func='mean'):
    """
    Aggregates values from a DataFrame for each month, latitude, longitude, and year,
    then reshapes the result into a 4D NumPy array.
    
    Parameters:
    - df: Pandas DataFrame with columns ['time', 'lat', 'lon', 'tp']
    - agg_func: String!tchyLlama65
     name of the aggregation function to use (e.g., 'mean', 'sum')
    
    Returns:
    - A 4D NumPy array with shape [12, 90, 40, -1], where -1 is the number of years
    """
    # Ensure 'time' is a datetime type
    df['time'] = pd.to_datetime(df['time'])
    
    # Extract month and year
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year - df['time'].dt.year.min()  # Normalize year
    
    # Group and aggregate
    aggregated_df = df.groupby(['month', 'lat', 'lon', 'year'])['tp'].agg(agg_func).reset_index()
    
    # Pivot
    pivoted_df = aggregated_df.pivot_table(index=['month', 'lat', 'lon'], columns='year', values='tp', fill_value=0)
    
    # Convert to a 4D array and reshape
    num_years = len(df['year'].unique())
    result_array = pivoted_df.values.reshape((12, 90, 40, num_years))
    
    return result_array

# Example usage:
# Assuming your DataFrame is named df and has the required columns
# result_array = aggregate_to_4d_array(df, agg_func='mean')
# Now result_array is the desired 4D array


def compute_kde_results(rcm_5d, weights_arr):
    """
    Computes Kernel Density Estimation (KDE) results using the provided data and weights.

    This function iterates over the specified dimensions of the input data arrays, repeats data 
    points according to their corresponding weights, applies a logarithmic transformation, and 
    computes the Gaussian KDE for each (i, j, k) combination.

    Parameters
    ----------
    rcm_5d : numpy.ndarray
        A 5-dimensional NumPy array of shape (R, I, J, K, L) containing the data.
        - R: Number of models or realizations.
        - I, J, K: Spatial or other indices over which to iterate.
        - L: Length of the data series for each model at each (i, j, k) point.

    weights_arr : numpy.ndarray
        A 5-dimensional NumPy array of shape (R, I, J, K, 1) containing the weights.
        The weights determine how many times each data point is repeated in the KDE computation.

    Returns
    -------
    kde_results : numpy.ndarray
        A 3-dimensional NumPy array of shape (I, J, K), where each element is a `gaussian_kde` 
        object representing the KDE computed for that (i, j, k) combination.

    Notes
    -----
    - The function ensures that weights are converted to integers, as they represent repetition counts.
    - Logarithmic transformation is applied to the concatenated data array before computing the KDE.
    - If there are no data points (e.g., all weights are zero), the KDE result is set to `None` for 
      that (i, j, k) combination.
    - The function prints the current indices (i, j, k) during computation for progress tracking.

    Examples
    --------
    >>> kde_results = compute_kde_results(rcm_5d, weights_arr)
    """

    import numpy as np
    from scipy.stats import gaussian_kde

    R, I, J, K, L = rcm_5d.shape

    # Initialize an empty array to store the KDE results
    kde_results = np.empty((I, J, K), dtype=object)

    for i in range(I):
        for j in range(J):
            for k in range(K):
                temp_list = []
                for r in range(R):
                    # Convert weight to integer repetition count
                    repeat_count = int(weights_arr[r, i, j, k, 0])
                    if repeat_count > 0:
                        data_array = rcm_5d[r, i, j, k, :]
                        # Repeat data points according to the weight
                        repeat_array = np.repeat(data_array, repeat_count)
                        temp_list.append(repeat_array)
                if temp_list:
                    # Concatenate all repeated data points
                    temp_array = np.concatenate(temp_list)
                    # Apply logarithmic transformation
                    temp_array = np.log(temp_array)
                    # Compute Gaussian KDE
                    temp_kde = gaussian_kde(temp_array)
                    kde_results[i, j, k] = temp_kde
                else:
                    kde_results[i, j, k] = None
                print(i, j, k)
    return kde_results

def compute_weighted_data_points(rcm_5d, weights_arr):
    """
    Computes weighted data points for each (i, j, k) combination without applying logarithmic transformation.

    This function iterates over the specified dimensions of the input data arrays, computes integer weights 
    that sum to 100 for each (i, j, k) combination, repeats data points according to these integer weights, 
    and collects the resulting data points into a 5-dimensional array.

    Parameters
    ----------
    rcm_5d : numpy.ndarray
        A 5-dimensional NumPy array of shape (R, I, J, K, L) containing the data.
        - R: Number of models or realizations.
        - I, J, K: Spatial or other indices over which to iterate.
        - L: Length of the data series for each model at each (i, j, k) point.

    weights_arr : numpy.ndarray
        A 5-dimensional NumPy array of shape (R, I, J, K, 1) containing the weights.
        The weights determine how the data points are weighted.
        The weights for each (i, j, k) sum to 100.

    Returns
    -------
    data_points_array : numpy.ndarray
        A 5-dimensional NumPy array of shape (I, J, K, 100, L), where for each (i, j, k) combination,
        the data points are repeated according to the integer weights and collected into an array of shape (100, L).

    Notes
    -----
    - The function computes integer weights that sum to 100 for each (i, j, k) combination.
    - No logarithmic transformation is applied to the data.
    - The function handles potential rounding issues when converting weights to integer counts.
    - The function prints the current indices (i, j, k) during computation for progress tracking.

    Examples
    --------
    >>> data_points_array = compute_weighted_data_points(rcm_5d, weights_arr)
    """

    import numpy as np

    R, I, J, K, L = rcm_5d.shape

    # Initialize an empty array to store the data points
    data_points_array = np.zeros((I, J, K, 100, L))

    for i in range(I):
        for j in range(J):
            for k in range(K):
                # Get the weights for this (i, j, k)
                weights = weights_arr[:, i, j, k, 0]  # Shape: (R,)

                # Use the weights directly as counts_floats
                counts_floats = weights.copy()

                # Take the floor of counts_floats to get counts_int
                counts_int = np.floor(counts_floats).astype(int)

                # Compute the sum of counts_int
                counts_sum = np.sum(counts_int)

                # Compute the difference from total (100)
                diff = int(100 - counts_sum)

                # Handle rounding errors
                fractional_parts = counts_floats - counts_int
                if diff > 0:
                    # Add one to the counts with the largest fractional parts
                    indices = np.argsort(-fractional_parts)  # Indices of largest fractions
                    counts_int[indices[:diff]] += 1
                elif diff < 0:
                    # Subtract one from the counts with the smallest fractional parts
                    indices = np.argsort(fractional_parts)  # Indices of smallest fractions
                    counts_int[indices[:abs(diff)]] -= 1

                # Verify that counts_int sums to 100
                total_weight = np.sum(counts_int)
                if total_weight != 100:
                    raise ValueError(f"Total weight at index ({i}, {j}, {k}) is {total_weight}, expected 100.")

                temp_list = []
                for r in range(R):
                    repeat_count = counts_int[r]
                    if repeat_count > 0:
                        data_array = rcm_5d[r, i, j, k, :]  # Shape: (L,)
                        # Repeat data points according to the integer weight
                        repeated_data = np.tile(data_array, (repeat_count, 1))  # Shape: (repeat_count, L)
                        temp_list.append(repeated_data)
                if temp_list:
                    # Concatenate all repeated data points
                    temp_array = np.vstack(temp_list)  # Shape: (100, L)
                    # Assign to the output array
                    data_points_array[i, j, k, :, :] = temp_array
                else:
                    # If no data points, fill with zeros or handle accordingly
                    data_points_array[i, j, k, :, :] = np.zeros((100, L))
                print(f"Processed indices: ({i}, {j}, {k})")
    return data_points_array

def log_transform(data, epsilon=1e-5):
    return np.log(data + epsilon)

def compute_kde(data):
    """
    Computes the Kernel Density Estimation (KDE) for each sample in a 4-dimensional array.
    
    Parameters:
    - data: A 4-dimensional NumPy array where the first three dimensions are used to 
            iterate over, and the last dimension contains the data samples for KDE.
    
    Returns:
    - rcms_kde: A 3-dimensional NumPy array of the same shape as the first three dimensions
                   of the input data, containing KDE objects for each sample.
    """
    # Initialize an empty array for KDE results with the same shape as the first three dimensions of data
    rcms_kde = np.empty(data.shape[:-1], dtype=object)
    
    # Iterate over the array
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                # Extract the vector
                sample = data[i, j, k, :]
                
                # Apply KDE
                kde = gaussian_kde(sample)
                
                # Store the KDE object
                rcms_kde[i, j, k] = kde
                
    return rcms_kde