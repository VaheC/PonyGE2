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
def max_min_diff(arr, window):

    diff = moving_max(arr, window) - moving_min(arr, window)

    for i in range(len(diff)):
        if diff[i] < 0:
            diff[i] = -1 * diff[i]
    
    return diff

@njit(cache=True)
def fibbo_prices(arr, window, fibbo_pct):

    max_p = moving_max(arr, window)

    min_p = moving_min(arr, window)

    diff = max_p - min_p

    for i in range(len(diff)):
        if diff[i] < 0:
            diff[i] = -1 * diff[i]

    fibbo_p = max_p - diff * fibbo_pct
    
    return fibbo_p

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

def get_exit_point(signal_condition, n_bars=7):
    temp_condition = signal_condition.copy()
    idx = np.where(temp_condition==True)[0]
    temp_condition[idx] = False
    idx = idx + n_bars + 1
    temp_condition[idx] = True
    return temp_condition
