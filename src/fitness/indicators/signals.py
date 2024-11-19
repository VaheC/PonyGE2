import numpy as np
from numba import njit

@njit(cache=True)
def moving_max(arr, window):
    # Initialize the result array with 999 for cases with insufficient data
    result = np.full(len(arr), np.nan, dtype=arr.dtype)
    
    for i in range(len(arr)):
        if i + 1 >= window:
            result[i] = np.max(arr[i + 1 - window:i + 1])
        # else:
        #     result[i] = 999  # Not enough data to calculate max

    return result

@njit(cache=True)
def moving_min(arr, window):
    # Initialize the result array with 999 for cases with insufficient data
    result = np.full(len(arr), np.nan, dtype=arr.dtype)
    
    for i in range(len(arr)):
        if i + 1 >= window:
            result[i] = np.min(arr[i + 1 - window:i + 1])
        # else:
        #     result[i] = 999  # Not enough data to calculate min

    return result

@njit(cache=True)
def moving_percentile(arr, window, percentile):
    # Initialize the result array with NaNs (or you can use 999 for insufficient data if needed)
    result = np.full(len(arr), np.nan, dtype=arr.dtype)
    
    for i in range(len(arr)):
        if i + 1 >= window:
            # Calculate percentile for the current window
            window_values = arr[i + 1 - window:i + 1]
            result[i] = np.percentile(window_values, percentile * 100)
        # else:
        #     result[i] = 999  # Not enough data to calculate the percentile

    return result

@njit(cache=True)
def moving_std(arr, window):
    # Initialize the result array with NaNs (or you can use 999 for insufficient data if needed)
    result = np.full(len(arr), np.nan, dtype=arr.dtype)
    
    for i in range(len(arr)):
        if i + 1 >= window:
            # Calculate percentile for the current window
            window_values = arr[i + 1 - window:i + 1]
            result[i] = np.std(window_values)

    return result
