import numpy as np
from numba import njit
import math
from  .numba_indicators_nan import *
from performance.helper_func import get_lag

@njit(cache=True)
def moving_average_group(series, window):
    result = np.full(series.shape, np.nan)
    for i in range(series.shape[1]):
        result[:, i] = moving_average(series[:, i], window)
    return result

@njit(cache=True)
def exponential_moving_average_group(series, window):
    result = np.full(series.shape, np.nan)
    for i in range(series.shape[1]):
        result[:, i] = exponential_moving_average(series[:, i], window)
    return result

@njit(cache=True)
def relative_strength_index_group(series, window=14):
    result = np.full(series.shape, np.nan)
    for i in range(series.shape[1]):
        result[:, i] = relative_strength_index(series[:, i], window)
    return result

@njit(cache=True)
def macd_line_group(series, short_window=12, long_window=26):
    result = np.full(series.shape, np.nan)
    for i in range(series.shape[1]):
        result[:, i] = macd_line(series[:, i], short_window, long_window)
    return result

@njit(cache=True)
def macd_group(series, short_window=12, long_window=26, signal_window=9):
    macd_mat = np.full(series.shape, np.nan)
    signal_mat = np.full(series.shape, np.nan)
    hist_mat = np.full(series.shape, np.nan)
    for i in range(series.shape[1]):
        macd_col, signal_col, hist_col = macd(series[:, i], short_window, long_window, signal_window)
        macd_mat[:, i] = macd_col
        signal_mat[:, i] = signal_col
        hist_mat[:, i] = hist_col
    return macd_mat, signal_mat, hist_mat

@njit(cache=True)
def bollinger_bands_group(series, window=20, num_std_dev=2):
    ma = np.full(series.shape, np.nan)
    upper = np.full(series.shape, np.nan)
    lower = np.full(series.shape, np.nan)
    for i in range(series.shape[1]):
        ma_col, upper_col, lower_col = bollinger_bands(series[:, i], window, num_std_dev)
        ma[:, i] = ma_col
        upper[:, i] = upper_col
        lower[:, i] = lower_col
    return ma, upper, lower

@njit(cache=True)
def momentum_group(series, window):
    result = np.full(series.shape, np.nan)
    for i in range(series.shape[1]):
        result[:, i] = momentum(series[:, i], window)
    return result

@njit(cache=True)
def stochastic_oscillator_group(series, window=14):
    result = np.full(series.shape, np.nan)
    for i in range(series.shape[1]):
        result[:, i] = stochastic_oscillator(series[:, i], window)
    return result

@njit(cache=True)
def true_range_group(high, low, close):
    result = np.zeros(close.shape)
    for i in range(close.shape[1]):
        result[:, i] = true_range(high[:, i], low[:, i], close[:, i])
    return result

@njit(cache=True)
def average_true_range_group(high, low, close, window=14):
    result = np.full(close.shape, np.nan)
    for i in range(close.shape[1]):
        result[:, i] = average_true_range(high[:, i], low[:, i], close[:, i], window)
    return result

@njit(cache=True)
def moving_average_difference_group(series, short_window, long_window):
    result = np.full(series.shape, np.nan)
    for i in range(series.shape[1]):
        result[:, i] = moving_average_difference(series[:, i], short_window, long_window)
    return result

@njit(cache=True)
def linear_perc_atr_group(high, low, close, window=14):
    result = np.full(close.shape, np.nan)
    for i in range(close.shape[1]):
        result[:, i] = linear_perc_atr(high[:, i], low[:, i], close[:, i], window)
    return result

@njit(cache=True)
def quadratic_perc_atr_group(high, low, close, window=14):
    result = np.full(close.shape, np.nan)
    for i in range(close.shape[1]):
        result[:, i] = quadratic_perc_atr(high[:, i], low[:, i], close[:, i], window)
    return result

@njit(cache=True)
def cubic_perc_atr_group(high, low, close, window=14):
    result = np.full(close.shape, np.nan)
    for i in range(close.shape[1]):
        result[:, i] = cubic_perc_atr(high[:, i], low[:, i], close[:, i], window)
    return result

@njit(cache=True)
def adx_group(high, low, close, window=14):
    result = np.full(close.shape, np.nan)
    for i in range(close.shape[1]):
        result[:, i] = adx(high[:, i], low[:, i], close[:, i], window)
    return result

@njit(cache=True)
def min_max_adx_group(adx_matrix, window=14):
    min_adx_matrix = np.full(adx_matrix.shape, np.nan)
    max_adx_matrix = np.full(adx_matrix.shape, np.nan)
    for i in range(adx_matrix.shape[1]):
        min_adx_matrix[:, i], max_adx_matrix[:, i] = min_max_adx(adx_matrix[:, i], window)
    return min_adx_matrix, max_adx_matrix

@njit(cache=True)
def residual_min_max_adx_group(adx_matrix, window=14):
    residual_min = np.full(adx_matrix.shape, np.nan)
    residual_max = np.full(adx_matrix.shape, np.nan)
    for i in range(adx_matrix.shape[1]):
        residual_min[:, i], residual_max[:, i] = residual_min_max_adx(adx_matrix[:, i], window)
    return residual_min, residual_max

@njit(cache=True)
def delta_accel_adx_group(adx_matrix, window=14):
    delta_matrix = np.full(adx_matrix.shape, np.nan)
    accel_matrix = np.full(adx_matrix.shape, np.nan)
    for i in range(adx_matrix.shape[1]):
        delta_matrix[:, i], accel_matrix[:, i] = delta_accel_adx(adx_matrix[:, i], window)
    return delta_matrix, accel_matrix

@njit(cache=True)
def intraday_intensity_group(high, low, close, volume):
    result = np.full(close.shape, np.nan)
    for i in range(close.shape[1]):
        result[:, i] = intraday_intensity(high[:, i], low[:, i], close[:, i], volume[:, i])
    return result

@njit(cache=True)
def delta_intraday_intensity_group(high, low, close, volume):
    res = np.empty_like(close)
    for col in range(close.shape[1]):
        res[:, col] = delta_intraday_intensity(high[:, col], low[:, col], close[:, col], volume[:, col])
    return res

@njit(cache=True)
def reactivity_group(prices, window=14):
    res = np.empty_like(prices)
    for col in range(prices.shape[1]):
        res[:, col] = reactivity(prices[:, col], window)
    return res

@njit(cache=True)
def delta_reactivity_group(reactivity):
    res = np.empty_like(reactivity)
    for col in range(reactivity.shape[1]):
        res[:, col] = delta_reactivity(reactivity[:, col])
    return res

@njit(cache=True)
def min_reactivity_group(reactivity, window=14):
    res = np.empty_like(reactivity)
    for col in range(reactivity.shape[1]):
        res[:, col] = min_reactivity(reactivity[:, col], window)
    return res

@njit(cache=True)
def max_reactivity_group(reactivity, window=14):
    res = np.empty_like(reactivity)
    for col in range(reactivity.shape[1]):
        res[:, col] = max_reactivity(reactivity[:, col], window)
    return res

@njit(cache=True)
def close_to_close_group(close):
    res = np.empty_like(close)
    for col in range(close.shape[1]):
        res[:, col] = close_to_close(close[:, col])
    return res

@njit(cache=True)
def n_day_high_group(prices, n):
    res = np.empty_like(prices)
    for col in range(prices.shape[1]):
        res[:, col] = n_day_high(prices[:, col], n)
    return res

@njit(cache=True)
def n_day_low_group(prices, n):
    res = np.empty_like(prices)
    for col in range(prices.shape[1]):
        res[:, col] = n_day_low(prices[:, col], n)
    return res

@njit(cache=True)
def close_minus_moving_average_group(close, window):
    res = np.empty_like(close)
    for col in range(close.shape[1]):
        res[:, col] = close_minus_moving_average(close[:, col], window)
    return res

@njit(cache=True)
def linear_deviation_group(prices, window):
    res = np.empty_like(prices)
    for col in range(prices.shape[1]):
        res[:, col] = linear_deviation(prices[:, col], window)
    return res

@njit(cache=True)
def quadratic_deviation_group(prices, window):
    res = np.empty_like(prices)
    for col in range(prices.shape[1]):
        res[:, col] = quadratic_deviation(prices[:, col], window)
    return res

@njit(cache=True)
def cubic_deviation_group(prices, window):
    res = np.empty_like(prices)
    for col in range(prices.shape[1]):
        res[:, col] = cubic_deviation(prices[:, col], window)
    return res

@njit(cache=True)
def abs_price_change_oscillator_group(close, window):
    res = np.empty_like(close)
    for col in range(close.shape[1]):
        res[:, col] = abs_price_change_oscillator(close[:, col], window)
    return res

@njit(cache=True)
def delta_intraday_intensity_group(high, low, close, volume):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = delta_intraday_intensity(high[:, i], low[:, i], close[:, i], volume[:, i])
    return out

@njit(cache=True)
def reactivity_group(prices, window=14):
    out = np.empty_like(prices)
    for i in range(prices.shape[1]):
        out[:, i] = reactivity(prices[:, i], window)
    return out

@njit(cache=True)
def delta_reactivity_group(reactivity_values):
    out = np.empty_like(reactivity_values)
    for i in range(reactivity_values.shape[1]):
        out[:, i] = delta_reactivity(reactivity_values[:, i])
    return out

@njit(cache=True)
def min_reactivity_group(reactivity_values, window=14):
    out = np.empty_like(reactivity_values)
    for i in range(reactivity_values.shape[1]):
        out[:, i] = min_reactivity(reactivity_values[:, i], window)
    return out

@njit(cache=True)
def max_reactivity_group(reactivity_values, window=14):
    out = np.empty_like(reactivity_values)
    for i in range(reactivity_values.shape[1]):
        out[:, i] = max_reactivity(reactivity_values[:, i], window)
    return out

@njit(cache=True)
def close_to_close_group(close):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = close_to_close(close[:, i])
    return out

@njit(cache=True)
def atr_ratio_group(high, low, close, window=14):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = atr_ratio(high[:, i], low[:, i], close[:, i], window)
    return out

@njit(cache=True)
def n_day_narrower_wider_group(high, low, n):
    narrower_out = np.empty_like(high)
    wider_out = np.empty_like(high)
    for i in range(high.shape[1]):
        narrower, wider = n_day_narrower_wider(high[:, i], low[:, i], n)
        narrower_out[:, i] = narrower
        wider_out[:, i] = wider
    return narrower_out, wider_out

@njit(cache=True)
def price_skewness_group(prices, window):
    out = np.empty_like(prices)
    for i in range(prices.shape[1]):
        out[:, i] = price_skewness(prices[:, i], window)
    return out

@njit(cache=True)
def change_skewness_group(prices, window):
    out = np.empty_like(prices)
    for i in range(prices.shape[1]):
        out[:, i] = change_skewness(prices[:, i], window)
    return out

@njit(cache=True)
def price_kurtosis_group(prices, window):
    out = np.empty_like(prices)
    for i in range(prices.shape[1]):
        out[:, i] = price_kurtosis(prices[:, i], window)
    return out

@njit(cache=True)
def change_kurtosis_group(prices, window):
    out = np.empty_like(prices)
    for i in range(prices.shape[1]):
        out[:, i] = change_kurtosis(prices[:, i], window)
    return out

@njit(cache=True)
def delta_price_skewness_group(price_skewness_values):
    out = np.empty_like(price_skewness_values)
    for i in range(price_skewness_values.shape[1]):
        out[:, i] = delta_price_skewness(price_skewness_values[:, i])
    return out

@njit(cache=True)
def delta_change_skewness_group(change_skewness_values):
    out = np.empty_like(change_skewness_values)
    for i in range(change_skewness_values.shape[1]):
        out[:, i] = delta_change_skewness(change_skewness_values[:, i])
    return out

@njit(cache=True)
def delta_intraday_intensity_group(high, low, close, volume):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = delta_intraday_intensity(high[:, i], low[:, i], close[:, i], volume[:, i])
    return out

@njit(cache=True)
def reactivity_group(prices, window=14):
    out = np.empty_like(prices)
    for i in range(prices.shape[1]):
        out[:, i] = reactivity(prices[:, i], window)
    return out

@njit(cache=True)
def delta_reactivity_group(reactivity_values):
    out = np.empty_like(reactivity_values)
    for i in range(reactivity_values.shape[1]):
        out[:, i] = delta_reactivity(reactivity_values[:, i])
    return out

@njit(cache=True)
def min_reactivity_group(reactivity_values, window=14):
    out = np.empty_like(reactivity_values)
    for i in range(reactivity_values.shape[1]):
        out[:, i] = min_reactivity(reactivity_values[:, i], window)
    return out

@njit(cache=True)
def max_reactivity_group(reactivity_values, window=14):
    out = np.empty_like(reactivity_values)
    for i in range(reactivity_values.shape[1]):
        out[:, i] = max_reactivity(reactivity_values[:, i], window)
    return out

@njit(cache=True)
def close_to_close_group(close):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = close_to_close(close[:, i])
    return out

@njit(cache=True)
def delta_price_kurtosis_group(price_kurtosis_values):
    out = np.empty_like(price_kurtosis_values)
    for i in range(price_kurtosis_values.shape[1]):
        out[:, i] = delta_price_kurtosis(price_kurtosis_values[:, i])
    return out

@njit(cache=True)
def delta_change_kurtosis_group(change_kurtosis_values):
    out = np.empty_like(change_kurtosis_values)
    for i in range(change_kurtosis_values.shape[1]):
        out[:, i] = delta_change_kurtosis(change_kurtosis_values[:, i])
    return out

@njit(cache=True)
def volume_momentum_group(volume, window):
    out = np.empty_like(volume)
    for i in range(volume.shape[1]):
        out[:, i] = volume_momentum(volume[:, i], window)
    return out

@njit(cache=True)
def delta_volume_momentum_group(volume_momentum_values):
    out = np.empty_like(volume_momentum_values)
    for i in range(volume_momentum_values.shape[1]):
        out[:, i] = delta_volume_momentum(volume_momentum_values[:, i])
    return out

@njit(cache=True)
def diff_volume_weighted_ma_over_ma_group(price, volume, window):
    out = np.empty_like(price)
    for i in range(price.shape[1]):
        out[:, i] = diff_volume_weighted_ma_over_ma(price[:, i], volume[:, i], window)
    return out

@njit(cache=True)
def custom_histogram_group(data, bins):
    n_rows, n_cols = data.shape
    hist_matrix = np.empty((n_cols, bins), dtype=np.int32)
    for i in range(n_cols):
        hist_matrix[i, :] = custom_histogram(data[:, i], bins)
    return hist_matrix

@njit(cache=True)
def price_entropy_group(prices, window, bins=10):
    out = np.empty_like(prices)
    for i in range(prices.shape[1]):
        out[:, i] = price_entropy(prices[:, i], window, bins)
    return out

@njit(cache=True)
def volume_entropy_group(volume, window, bins=10):
    out = np.empty_like(volume)
    for i in range(volume.shape[1]):
        out[:, i] = volume_entropy(volume[:, i], window, bins)
    return out

@njit(cache=True)
def calculate_moving_support_resistance_group(prices, window, lag):
    n_rows, n_cols = prices.shape
    support_group = np.empty_like(prices)
    resistance_group = np.empty_like(prices)
    for i in range(n_cols):
        support_group[:, i], resistance_group[:, i] = calculate_moving_support_resistance(prices[:, i], window, lag)
    return support_group, resistance_group

@njit(cache=True)
def chaikin_ad_line_group(close, low, high, volume):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = chaikin_ad_line(close[:, i], low[:, i], high[:, i], volume[:, i])
    return out

@njit(cache=True)
def chaikin_ad_oscillator_group(ad_line, short_window, long_window):
    out = np.empty_like(ad_line)
    for i in range(ad_line.shape[1]):
        out[:, i] = chaikin_ad_oscillator(ad_line[:, i], short_window, long_window)
    return out

@njit(cache=True)
def absolute_price_oscillator_group(close, short_window, long_window):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = absolute_price_oscillator(close[:, i], short_window, long_window)
    return out

@njit(cache=True)
def on_balance_volume_group(price, volume):
    out = np.empty_like(price)
    for i in range(price.shape[1]):
        out[:, i] = on_balance_volume(price[:, i], volume[:, i])
    return out

@njit(cache=True)
def delta_on_balance_volume_group(obv_values):
    out = np.empty_like(obv_values)
    for i in range(obv_values.shape[1]):
        out[:, i] = delta_on_balance_volume(obv_values[:, i])
    return out

@njit(cache=True)
def positive_volume_indicator_group(volume):
    out = np.empty_like(volume)
    for i in range(volume.shape[1]):
        out[:, i] = positive_volume_indicator(volume[:, i])
    return out

@njit(cache=True)
def negative_volume_indicator_group(volume):
    out = np.empty_like(volume)
    for i in range(volume.shape[1]):
        out[:, i] = negative_volume_indicator(volume[:, i])
    return out

@njit(cache=True)
def product_price_volume_group(price, volume):
    out = np.empty_like(price)
    for i in range(price.shape[1]):
        out[:, i] = product_price_volume(price[:, i], volume[:, i])
    return out

@njit(cache=True)
def sum_price_volume_group(price, volume, window):
    out = np.empty_like(price)
    for i in range(price.shape[1]):
        out[:, i] = sum_price_volume(price[:, i], volume[:, i], window)
    return out

@njit(cache=True)
def triple_exponential_moving_average_group(close, period):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = triple_exponential_moving_average(close[:, i], period)
    return out

@njit(cache=True)
def williams_r_group(high, low, close, period):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = williams_r(high[:, i], low[:, i], close[:, i], period)
    return out

@njit(cache=True)
def z_score_group(data):
    out = np.empty_like(data)
    for i in range(data.shape[1]):
        out[:, i] = z_score(data[:, i])
    return out

@njit(cache=True)
def rolling_drawdown_group(data, window_size):
    out = np.empty_like(data)
    for i in range(data.shape[1]):
        out[:, i] = rolling_drawdown(data[:, i], window_size)
    return out

@njit(cache=True)
def rolling_volatility_group(data, window_size):
    out = np.empty_like(data)
    for i in range(data.shape[1]):
        out[:, i] = rolling_volatility(data[:, i], window_size)
    return out

@njit(cache=True)
def rolling_parkinson_estimator_group(high, low, window_size):
    out = np.empty_like(high)
    for i in range(high.shape[1]):
        out[:, i] = rolling_parkinson_estimator(high[:, i], low[:, i], window_size)
    return out

@njit(cache=True)
def aroon_group(prices, period):
    n_rows, n_cols = prices.shape
    aroon_up_group = np.empty_like(prices)
    aroon_down_group = np.empty_like(prices)
    for i in range(n_cols):
        aroon_up_group[:, i], aroon_down_group[:, i] = aroon(prices[:, i], period)
    return aroon_up_group, aroon_down_group

@njit(cache=True)
def balance_of_power_group(close, high, low):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = balance_of_power(close[:, i], high[:, i], low[:, i])
    return out

@njit(cache=True)
def double_exponential_moving_average_group(close, period):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = double_exponential_moving_average(close[:, i], period)
    return out

@njit(cache=True)
def directional_movement_index_group(high, low, close, period):
    n_rows, n_cols = close.shape
    dmi_plus_group = np.empty_like(close)
    dmi_minus_group = np.empty_like(close)
    dx_group = np.empty_like(close)
    for i in range(n_cols):
        dmi_plus_group[:, i], dmi_minus_group[:, i], dx_group[:, i] = directional_movement_index(high[:, i], low[:, i], close[:, i], period)
    return dmi_plus_group, dmi_minus_group, dx_group

@njit(cache=True)
def hilbert_dominant_cycle_period_group(close, period):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = hilbert_dominant_cycle_period(close[:, i], period)
    return out

@njit(cache=True)
def hilbert_dominant_cycle_phase_group(close, period):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = hilbert_dominant_cycle_phase(close[:, i], period)
    return out

@njit(cache=True)
def hilbert_phasor_components_group(close):
    n_rows, n_cols = close.shape
    phasor_group = np.full((n_rows, n_cols, 2), np.nan)
    for i in range(n_cols):
        phasor_group[:, i, :] = hilbert_phasor_components(close[:, i])
    return phasor_group

@njit(cache=True)
def ultimate_oscillator_group(high_mat, low_mat, close_mat, period1, period2, period3):
    n, m = close_mat.shape
    out = np.full((n, m), np.nan)
    for j in range(m):
        out[:, j] = ultimate_oscillator(close_mat[:, j], low_mat[:, j], close_mat[:, j], period1, period2, period3)
    return out

@njit(cache=True)
def medprice_group(high_mat, low_mat):
    return (high_mat + low_mat) / 2

@njit(cache=True)
def ldecay_group(price_mat, period):
    n, m = price_mat.shape
    out = np.full((n, m), np.nan)
    for j in range(m):
        out[:, j] = ldecay(price_mat[:, j], period)
    return out

@njit(cache=True)
def logret_group(price_mat):
    n, m = price_mat.shape
    out = np.full((n, m), np.nan)
    for j in range(m):
        out[:, j] = logret(price_mat[:, j])
    return out

@njit(cache=True)
def pvi_group(price_mat, volume_mat):
    n, m = price_mat.shape
    out = np.full((n, m), np.nan)
    for j in range(m):
        out[:, j] = pvi(price_mat[:, j], volume_mat[:, j])
    return out

@njit(cache=True)
def hilbert_sinewave_group(close):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = hilbert_sinewave(close[:, i])
    return out

@njit(cache=True)
def hilbert_instantaneous_trendline_group(close):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = hilbert_instantaneous_trendline(close[:, i])
    return out

@njit(cache=True)
def hilbert_trend_vs_cycle_mode_group(close):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = hilbert_trend_vs_cycle_mode(close[:, i])
    return out

@njit(cache=True)
def mesa_adaptive_moving_average_group(close, period):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = mesa_adaptive_moving_average(close[:, i], period)
    return out

@njit(cache=True)
def median_price_group(high, low):
    out = np.empty_like(high)
    for i in range(high.shape[1]):
        out[:, i] = median_price(high[:, i], low[:, i])
    return out

@njit(cache=True)
def midpoint_over_period_group(close, period):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = midpoint_over_period(close[:, i], period)
    return out

@njit(cache=True)
def parabolic_sar_group(high, low):
    out = np.empty_like(high)
    for i in range(high.shape[1]):
        out[:, i] = parabolic_sar(high[:, i], low[:, i])
    return out

@njit(cache=True)
def delta_intraday_intensity_vec(high, low, close, volume):
    delta_intraday_intensity = np.full(len(close), np.nan)
    intraday_intensity = np.full(len(close), np.nan)
    for i in range(len(close)):
        high_low = high[i] - low[i]
        if high_low != 0:
            intraday_intensity[i] = ((close[i] - low[i]) - (high[i] - close[i])) / high_low * volume[i]
    for i in range(1, len(close)):
        if (not np.isnan(intraday_intensity[i])) and (not np.isnan(intraday_intensity[i - 1])):
            delta_intraday_intensity[i] = intraday_intensity[i] - intraday_intensity[i - 1]
    return delta_intraday_intensity

@njit(cache=True)
def delta_intraday_intensity_group(high, low, close, volume):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = delta_intraday_intensity_vec(high[:, i], low[:, i], close[:, i], volume[:, i])
    return out

@njit(cache=True)
def triple_exponential_moving_average_group(close, period):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = triple_exponential_moving_average(close[:, i], period)
    return out

@njit(cache=True)
def williams_r_group(high, low, close, period):
    out = np.empty_like(close)
    for i in range(close.shape[1]):
        out[:, i] = williams_r(high[:, i], low[:, i], close[:, i], period)
    return out

@njit(cache=True)
def z_score_group(data):
    out = np.empty_like(data)
    for i in range(data.shape[1]):
        out[:, i] = z_score(data[:, i])
    return out

@njit(cache=True)
def rolling_drawdown_group(data, window_size):
    out = np.empty_like(data)
    for i in range(data.shape[1]):
        out[:, i] = rolling_drawdown(data[:, i], window_size)
    return out

@njit(cache=True)
def rolling_volatility_group(data, window_size):
    out = np.empty_like(data)
    for i in range(data.shape[1]):
        out[:, i] = rolling_volatility(data[:, i], window_size)
    return out

@njit(cache=True)
def rolling_parkinson_estimator_group(high, low, window_size):
    out = np.empty_like(high)
    for i in range(high.shape[1]):
        out[:, i] = rolling_parkinson_estimator(high[:, i], low[:, i], window_size)
    return out

@njit(cache=True)
def rolling_rogers_satchell_estimator_group(open, high, low, close, window_size):
    out = np.empty_like(open)
    for i in range(open.shape[1]):
        out[:, i] = rolling_rogers_satchell_estimator(open[:, i], high[:, i], low[:, i], close[:, i], window_size)
    return out

@njit(cache=True)
def rolling_yang_zhang_estimator_group(open, high, low, close, window_size):
    out = np.empty_like(open)
    for i in range(open.shape[1]):
        out[:, i] = rolling_yang_zhang_estimator(open[:, i], high[:, i], low[:, i], close[:, i], window_size)
    return out

@njit(cache=True)
def rolling_garman_klass_estimator_group(open, high, low, close, window_size):
    out = np.empty_like(open)
    for i in range(open.shape[1]):
        out[:, i] = rolling_garman_klass_estimator(open[:, i], high[:, i], low[:, i], close[:, i], window_size)
    return out

@njit(cache=True)
def ultimate_oscillator_group(high_mat, low_mat, close_mat, period1, period2, period3):
    n, m = close_mat.shape
    out = np.full((n, m), np.nan)
    for j in range(m):
        out[:, j] = ultimate_oscillator(close_mat[:, j], low_mat[:, j], close_mat[:, j], period1, period2, period3)
    return out

@njit(cache=True)
def medprice_group(high_mat, low_mat):
    return (high_mat + low_mat) / 2

@njit(cache=True)
def ldecay_group(price_mat, period):
    n, m = price_mat.shape
    out = np.full((n, m), np.nan)
    for j in range(m):
        out[:, j] = ldecay(price_mat[:, j], period)
    return out

@njit(cache=True)
def logret_group(price_mat):
    n, m = price_mat.shape
    out = np.full((n, m), np.nan)
    for j in range(m):
        out[:, j] = logret(price_mat[:, j])
    return out

@njit(cache=True)
def pvi_group(price_mat, volume_mat):
    n, m = price_mat.shape
    out = np.full((n, m), np.nan)
    for j in range(m):
        out[:, j] = pvi(price_mat[:, j], volume_mat[:, j])
    return out

@njit(cache=True)
def pctret_group(price_mat):
    n, m = price_mat.shape
    out = np.full((n, m), np.nan)
    for j in range(m):
        out[:, j] = pctret(price_mat[:, j])
    return out

@njit(cache=True)
def cti_group(price_mat, period):
    n, m = price_mat.shape
    out = np.full((n, m), np.nan)
    for j in range(m):
        out[:, j] = cti(price_mat[:, j], period)
    return out

@njit(cache=True)
def dema_group(price_mat, period):
    n, m = price_mat.shape
    out = np.full((n, m), np.nan)
    for j in range(m):
        out[:, j] = dema(price_mat[:, j], period)
    return out

@njit(cache=True)
def hma_group(price_mat, period):
    n, m = price_mat.shape
    out = np.full((n, m), np.nan)
    for j in range(m):
        out[:, j] = hma(price_mat[:, j], period)
    return out

@njit(cache=True)
def stochastic_oscillator_kd_group(close_mat, high_mat, low_mat, period=14, smooth_period=3):
    n, m = close_mat.shape
    stoch_k_mat = np.full((n, m), np.nan)
    stoch_d_mat = np.full((n, m), np.nan)
    for j in range(m):
        k, d = stochastic_oscillator_kd(close_mat[:, j], high_mat[:, j], low_mat[:, j], period, smooth_period)
        stoch_k_mat[:, j] = k
        stoch_d_mat[:, j] = d
    return stoch_k_mat, stoch_d_mat

@njit(cache=True)
def money_flow_index_group(high_mat, low_mat, close_mat, volume_mat, period=14):
    n, m = close_mat.shape
    out = np.full((n, m), np.nan)
    for j in range(m):
        out[:, j] = money_flow_index(high_mat[:, j], low_mat[:, j], close_mat[:, j], volume_mat[:, j], period)
    return out

@njit(cache=True)
def calculate_keltner_channels_group(high_mat, low_mat, close_mat, period=20, multiplier=2):
    n, m = close_mat.shape
    middle = np.full((n, m), np.nan)
    upper = np.full((n, m), np.nan)
    lower = np.full((n, m), np.nan)
    for j in range(m):
        mid, up, low = calculate_keltner_channels(high_mat[:, j], low_mat[:, j], close_mat[:, j], period, multiplier)
        middle[:, j] = mid
        upper[:, j] = up
        lower[:, j] = low
    return middle, upper, lower

@njit(cache=True)
def commodity_channel_index_group(high_mat, low_mat, close_mat, period=20):
    n, m = close_mat.shape
    out = np.full((n, m), np.nan)
    for j in range(m):
        out[:, j] = commodity_channel_index(high_mat[:, j], low_mat[:, j], close_mat[:, j], period)
    return out

@njit(cache=True)
def bull_bar_tail_rolling_group(close_mat, open_mat, high_mat, low_mat, period=20):
    n, m = close_mat.shape
    out = np.full((n, m), np.nan, dtype=np.float64)
    for j in range(m):
        out[:, j] = bull_bar_tail_rolling(close_mat[:, j], open_mat[:, j], high_mat[:, j], low_mat[:, j], period)
    return out

@njit(cache=True)
def bear_bar_tail_rolling_group(close_mat, open_mat, high_mat, low_mat, period=20):
    n, m = close_mat.shape
    out = np.full((n, m), np.nan, dtype=np.float64)
    for j in range(m):
        out[:, j] = bear_bar_tail_rolling(close_mat[:, j], open_mat[:, j], high_mat[:, j], low_mat[:, j], period)
    return out

@njit(cache=True)
def awesome_oscillator_group(high_mat, low_mat, short_period=5, long_period=34):
    n, m = high_mat.shape
    out = np.full((n, m), np.nan)
    for j in range(m):
        out[:, j] = awesome_oscillator(high_mat[:, j], low_mat[:, j], short_period, long_period)
    return out

@njit(cache=True)
def rolling_max_index_group(arr_mat, window):
    n, m = arr_mat.shape
    out = np.full((n, m), np.nan, dtype=np.float64)
    for j in range(m):
        out[:, j] = rolling_max_index(arr_mat[:, j], window)
    return out

@njit(cache=True)
def rolling_min_index_group(arr_mat, window):
    n, m = arr_mat.shape
    out = np.full((n, m), np.nan, dtype=np.float64)
    for j in range(m):
        out[:, j] = rolling_min_index(arr_mat[:, j], window)
    return out

@njit(cache=True)
def get_lag_group(series, lag=1):
    result = np.full(series.shape, np.nan)
    for i in range(series.shape[1]):
        result[:, i] = get_lag(series[:, i], lag)
    return result