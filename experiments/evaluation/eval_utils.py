import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

def sample_moe(samples: np.ndarray, weights: np.ndarray, num_moe_samples: int) -> np.ndarray:
    """
    Generates samples from a Mixture of Experts (MoE) distribution.
    
    Parameters:
    ----------
    samples : np.ndarray
        Array of pre-generated samples for each expert across grid points.
        Shape: [E, D1, D2, ..., Dk, S]
            E: Number of experts
            D1, D2, ..., Dk: Grid dimensions
            S: Number of samples per expert at each grid point
    
    weights : np.ndarray
        Weights for each expert at each grid point.
        Shape: [E, D1, D2, ..., Dk]
            E: Number of experts (must match the first dimension of `samples`)
            D1, D2, ..., Dk: Grid dimensions (must match the corresponding dimensions in `samples`)
        Notes:
            - Weights should be non-negative.
            - Weights for each grid point should sum to 1 across the expert axis.
    
    num_moe_samples : int
        Number of samples to generate from the MoE distribution for each grid point.
    
    Returns:
    -------
    moe_samples : np.ndarray
        Generated MoE samples for each grid point.
        Shape: [D1, D2, ..., Dk, num_moe_samples]
    
    Raises:
    ------
    ValueError:
        If input dimensions do not align or weights do not sum to 1.
    """
    # Validate inputs
    if samples.ndim < 2:
        raise ValueError("`samples` array must have at least two dimensions (experts and samples).")
    
    if weights.ndim != samples.ndim - 1:
        raise ValueError("`weights` array must have one fewer dimension than `samples` array.")
    
    E_samples = samples.shape[0]
    E_weights = weights.shape[0]
    
    if E_samples != E_weights:
        raise ValueError(f"Number of experts in `samples` ({E_samples}) and `weights` ({E_weights}) must match.")
    
    # Check that weights are non-negative
    if np.any(weights < 0):
        raise ValueError("All weights must be non-negative.")
    
    # Check that weights sum to 1 across the expert axis
    weight_sums = weights.sum(axis=0)
    if not np.allclose(weight_sums, 1.0):
        raise ValueError("Weights must sum to 1 across the expert axis for each grid point.")
    
    # Extract grid dimensions and other parameters
    grid_shape = weights.shape[1:]  # [D1, D2, ..., Dk]
    num_grid_points = np.prod(grid_shape)
    num_experts = E_samples
    num_available_samples = samples.shape[-1]  # S
    
    # Reshape samples to [E, num_grid_points, S]
    samples_flat = samples.reshape(num_experts, num_grid_points, num_available_samples)
    
    # Reshape weights to [E, num_grid_points] and transpose to [num_grid_points, E]
    weights_flat = weights.reshape(num_experts, num_grid_points).T  # Shape: [num_grid_points, E]
    
    # Compute cumulative weights for each grid point
    cumulative_weights = np.cumsum(weights_flat, axis=1)  # Shape: [num_grid_points, E]
    
    # Generate random numbers for expert selection: [num_grid_points, num_moe_samples]
    random_values = np.random.rand(num_grid_points, num_moe_samples)
    
    # Vectorized expert selection using comparison and argmax
    # Expand dimensions to [num_grid_points, num_moe_samples, E] for broadcasting
    # Compare random_values with cumulative_weights to create a boolean mask
    # The first True in each row indicates the selected expert
    mask = random_values[:, :, np.newaxis] < cumulative_weights[:, np.newaxis, :]  # Shape: [num_grid_points, num_moe_samples, E]
    
    # Use argmax to find the first True value along the expert axis
    selected_experts = mask.argmax(axis=2)  # Shape: [num_grid_points, num_moe_samples]
    
    # Generate random sample indices from the selected experts: [num_grid_points, num_moe_samples]
    sample_indices = np.random.randint(0, num_available_samples, size=(num_grid_points, num_moe_samples))
    
    # Prepare indices for advanced indexing
    # Flatten the arrays to 1D for efficient indexing
    flat_selected_experts = selected_experts.flatten()          # Shape: [num_grid_points * num_moe_samples]
    flat_grid_cell_indices = np.repeat(np.arange(num_grid_points), num_moe_samples)  # Shape: [num_grid_points * num_moe_samples]
    flat_sample_indices = sample_indices.flatten()              # Shape: [num_grid_points * num_moe_samples]
    
    # Gather the selected samples using advanced indexing
    # `samples_flat` has shape [E, num_grid_points, S]
    # We index it as [selected_experts, grid_cells, sample_indices]
    moe_samples_flat = samples_flat[
        flat_selected_experts,      # Expert indices [num_grid_points * num_moe_samples]
        flat_grid_cell_indices,     # Grid cell indices [num_grid_points * num_moe_samples]
        flat_sample_indices         # Sample indices [num_grid_points * num_moe_samples]
    ]  # Shape: [num_grid_points * num_moe_samples]
    
    # Reshape the flat MoE samples back to [num_grid_points, num_moe_samples]
    moe_samples_flat = moe_samples_flat.reshape(num_grid_points, num_moe_samples)  # Shape: [num_grid_points, num_moe_samples]
    
    # Reshape back to the original grid dimensions with the new sample axis
    moe_samples = moe_samples_flat.reshape(*grid_shape, num_moe_samples)  # Shape: [D1, D2, ..., Dk, num_moe_samples]
    
    return moe_samples

def compute_inverse_weights(wasserstein_distances: np.ndarray) -> np.ndarray:
    """
    Compute weights inversely proportional to Wasserstein distances.
    The weights sum to 1 along the first axis (axis=0).

    Parameters:
    - wasserstein_distances (np.ndarray): 
        Array of Wasserstein distances with shape [5, 12, 180, 80].

    Returns:
    - weights (np.ndarray): 
        Array of weights with the same shape [5, 12, 180, 80],
        where weights are inversely proportional to distances and 
        sum to 1 along the first axis.
    """
    if wasserstein_distances.ndim != 4:
        raise ValueError(f"Expected a 4D array, but got {wasserstein_distances.ndim}D array.")
    
    if wasserstein_distances.shape[0] != 5:
        raise ValueError(f"Expected the first dimension to be of size 5, but got {wasserstein_distances.shape[0]}.")
    
    # Convert distances to float to ensure proper division
    distances = wasserstein_distances.astype(float)
    
    # Initialize weights array with zeros
    weights = np.zeros_like(distances)
    
    # Create a mask where distances are zero
    zero_mask = distances == 0  # Shape: [5, 12, 180, 80]
    
    # Count the number of zero distances along the first axis for each [12, 180, 80] position
    zero_counts = zero_mask.sum(axis=0)  # Shape: [12, 180, 80]
    
    # Identify positions where at least one distance is zero
    any_zero = zero_counts > 0  # Shape: [12, 180, 80]
    
    # **Case 1**: Positions with one or more zero distances
    if np.any(any_zero):
        # For positions with any zero distances:
        # Assign equal weights to all elements with zero distance
        # and zero weights to elements with non-zero distances
        # Expand zero_counts to broadcast correctly
        # To prevent division by zero, ensure zero_counts > 0
        weights[:, any_zero] = zero_mask[:, any_zero].astype(float)
        weights[:, any_zero] /= zero_counts[None, any_zero]
    
    # **Case 2**: Positions with no zero distances
    non_zero = ~any_zero  # Shape: [12, 180, 80]
    if np.any(non_zero):
        # Compute inverse distances
        inv_dist = 1.0 / distances[:, non_zero]  # Shape: [5, N], where N is number of non-zero positions
        
        # Sum of inverse distances for normalization
        sum_inv_dist = inv_dist.sum(axis=0)  # Shape: [N]
        
        # Normalize inverse distances to get weights
        weights[:, non_zero] = inv_dist / sum_inv_dist[None, :]
    
    return weights


def tile_with_neighbors_stride(data, window_size=3, fill_value=0):
    """
    Tile the data with its adjacent neighbors using sliding_window_view.

    Parameters:
    - data (np.ndarray): Input array with shape (months, lat, lon, years).
    - window_size (int): Size of the window (must be an odd integer >=1).
    - fill_value (float, optional): Value to assign to regions outside the boundaries. Defaults to 0.

    Returns:
    - tiled_data (np.ndarray): Output array with shape 
                               (months, lat, lon, years * window_size^2).
    """
    if data.ndim != 4:
        raise ValueError("Input data must be a 4D array with shape (months, lat, lon, years).")
    
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer greater than or equal to 1.")
    
    months, lat, lon, years = data.shape
    pad_size = window_size // 2

    # Pad the array
    padded_data = np.pad(
        data,
        pad_width=((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)),
        mode='constant',
        constant_values=fill_value
    )

    # Create sliding windows
    # sliding_window_view will create a view with a new set of dimensions for the window
    windows = sliding_window_view(padded_data, window_shape=(window_size, window_size), axis=(1, 2))
    # windows shape: (months, lat, lon, window_size, window_size, years)

    # Rearrange axes to bring window dimensions next to years
    windows = windows.transpose(0, 1, 2, 5, 3, 4)
    # windows shape: (months, lat, lon, years, window_size, window_size)

    # Reshape to combine window dimensions
    tiled_data = windows.reshape(months, lat, lon, years * window_size**2)

    return tiled_data

def df_to_4d_array(df, variable = 'mean'):
    """
    Converts a DataFrame with columns ['month', 'lon', 'lat', 'value'] to a 4D NumPy array.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame containing columns 'month', 'lon', 'lat', 'value'.

    Returns:
    -------
    np.ndarray
        A 4D NumPy array with dimensions [month, lon, lat, values].
    """
    # replaces NaN values with -9999
    df = df.fillna(-9999)

    # Pivot the DataFrame to create a 3D structure
    pivot_df = df.pivot_table(index='month', columns=['lon', 'lat'], values=variable, aggfunc='first')

    # Sort the pivot table
    pivot_df = pivot_df.sort_index(axis=1)  # Sort by 'lon' and 'lat'
    
    # Get the unique counts for each dimension
    months = df['month'].nunique()
    lons = df['lon'].nunique()
    lats = df['lat'].nunique()

    # Reshape the pivoted data into a 4D array
    array = pivot_df.values.reshape(months, lons, lats)

    # Replace -9999 with NaN
    array[array == -9999] = np.nan

    return array

import numpy as np
import pandas as pd

def aggregate_to_4d_array(df, value_col, agg_func='mean'):
    """
    Aggregates values from a DataFrame for each month, latitude, and longitude,
    then reshapes the result into a 4D NumPy array.

    Parameters:
    ----------
    - df: Pandas DataFrame with columns ['month', 'lat', 'lon', value_col]
    - value_col: String, name of the column to aggregate (e.g., 'tp')
    - agg_func: String, name of the aggregation function to use (e.g., 'mean', 'sum')

    Returns:
    -------
    - A 4D NumPy array with shape [num_months, num_lats, num_lons, num_values]
    
    Raises:
    -------
    - ValueError: If the number of entries (e.g., years) is inconsistent across lat, lon, and month groups.
    """
    
    # Group by lat, lon, and month and count the occurrences to check consistency
    value_counts = df.groupby(['lat', 'lon', 'month']).size().reset_index(name='counts')
    num_values = value_counts['counts'].nunique()  # Check if all groups have the same number of entries

    if num_values != 1:
        raise ValueError("The number of values is not consistent across all lat, lon, and month groups.")

    # Group and aggregate
    aggregated_df = df.groupby(['month', 'lat', 'lon'])[value_col].agg(agg_func).reset_index()

    # Get the number of unique months, lats, lons
    num_months = df['month'].nunique()
    num_lats = df['lat'].nunique()
    num_lons = df['lon'].nunique()
    num_values = value_counts['counts'].max()

    # Pivot table to structure data into the required format
    pivoted_df = aggregated_df.pivot_table(index=['month', 'lat', 'lon'], values=value_col, aggfunc='first', fill_value=np.nan)
    
    # Convert to a 4D array
    result_array = pivoted_df.values.reshape((num_months, num_lats, num_lons, num_values))

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