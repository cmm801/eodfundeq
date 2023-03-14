import numpy as np


def ffill(arr):
    """Method to front-fill NaNs in a numpy array."""
    mask = np.isnan(arr)

    # Get a matrix with index of valid data point, and 0 otherwise
    idx = np.where(~mask, np.arange(mask.shape[0]).reshape(-1, 1), 0)

    # Adjuste index so it contains the row index of the last non-NaN point
    np.maximum.accumulate(idx, axis=0, out=idx)

    # Add additional mask so NaNs beyond last valid data point don't get filled
    row_max = np.max(idx, axis=0)
    row_num = np.arange(idx.shape[0]).reshape(-1, 1)
    traing_nan_mask = (idx == row_max) & (idx < row_num)
    mask = mask & ~traing_nan_mask

    # Replace non-trailing NaNs with previous value
    arr[mask] = arr[idx[mask], np.nonzero(mask)[1]]
    return arr    

def rolling_sum(arr, n_periods, min_obs=None, fillna=True):
    """Compute a rolling sum of a numpy array."""
    if min_obs is None:
        min_obs = n_periods

    cs = np.nancumsum(arr, axis=0)
    roll_sum = np.vstack([
            np.nan * np.ones((
            n_periods-1, arr.shape[1]), dtype=arr.dtype),
            cs[n_periods-1, :],
            cs[n_periods:, :] - cs[:arr.shape[0]-n_periods, :]
    ])
    
    observations = np.ones_like(cs, dtype=int) 
    observations[np.isnan(arr)] = 0
    cs_obs_window = np.nancumsum(observations, axis=0)
    obs_in_window = np.vstack([
            np.nan * np.ones((
            n_periods-1, arr.shape[1]), dtype=arr.dtype),
            cs_obs_window[n_periods-1, :],
            cs_obs_window[n_periods:, :] - cs_obs_window[:arr.shape[0]-n_periods, :]
    ])
    roll_sum[obs_in_window < min_obs] = np.nan
    
    # Re-fill values that were originally NaNs
    if not fillna:
        roll_sum[np.isnan(arr)] = np.nan
    return roll_sum
