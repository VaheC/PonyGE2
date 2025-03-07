import numpy as np
from numba import njit
import math

# Moving Average (MA)
@njit(cache=True)
def moving_average(prices, window):
    ma = np.full(len(prices), np.nan)
    for i in range(len(prices) - window + 1):
        ma[i + window - 1] = np.mean(prices[i:i + window])
    return ma

# Exponential Moving Average (EMA)
@njit(cache=True)
def exponential_moving_average(prices, window):
    ema = np.full(len(prices), np.nan)
    alpha = 2 / (window + 1)
    ema[window - 1] = prices[0]
    for i in range(window, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema

# Relative Strength Index (RSI)
@njit(cache=True)
def relative_strength_index(prices, window=14):
    # prices = prices.flatten()
    rsi = np.full(len(prices), np.nan)
    gains = np.zeros(len(prices))
    losses = np.zeros(len(prices))
    
    for i in range(1, len(prices)):
        delta = prices[i] - prices[i - 1]
        if delta > 0:
            gains[i] = delta
        if delta < 0:
            losses[i] = -delta

    avg_gain = np.mean(gains[1:window+1])
    avg_loss = np.mean(losses[1:window+1])
    
    for i in range(window, len(prices)):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi

# Moving Average Convergence Divergence (MACD)
@njit(cache=True)
def macd(prices, short_window=12, long_window=26, signal_window=9):
    macd_line = np.full(len(prices), np.nan)
    signal_line = np.full(len(prices), np.nan)
    histogram = np.full(len(prices), np.nan)
    
    ema_short = exponential_moving_average(prices, short_window)
    ema_long = exponential_moving_average(prices, long_window)
    
    for i in range(long_window - 1, len(prices)):
        macd_line[i] = ema_short[i] - ema_long[i]

    ema_signal = exponential_moving_average(macd_line, signal_window)
    for i in range(long_window - 1, len(prices)):
        if not np.isnan(ema_signal[i]):
            signal_line[i] = ema_signal[i]
            histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram

# Moving Average Convergence Divergence (MACD)
@njit(cache=True)
def macd_line(prices, short_window=12, long_window=26):
    macd_line = np.full(len(prices), np.nan)
    
    ema_short = exponential_moving_average(prices, short_window)
    ema_long = exponential_moving_average(prices, long_window)
    
    for i in range(long_window - 1, len(prices)):
        macd_line[i] = ema_short[i] - ema_long[i]
    
    return macd_line

# Bollinger Bands (BB)
@njit(cache=True)
def bollinger_bands(prices, window=20, num_std_dev=2):
    ma = np.full(len(prices), np.nan)
    upper_band = np.full(len(prices), np.nan)
    lower_band = np.full(len(prices), np.nan)
    
    for i in range(len(prices) - window + 1):
        ma[i + window - 1] = np.mean(prices[i:i + window])
        std_dev = np.std(prices[i:i + window])
        upper_band[i + window - 1] = ma[i + window - 1] + (std_dev * num_std_dev)
        lower_band[i + window - 1] = ma[i + window - 1] - (std_dev * num_std_dev)
    
    return ma, upper_band, lower_band

# Momentum Indicator
@njit(cache=True)
def momentum(prices, window):
    momentum = np.full(len(prices), np.nan)
    for i in range(window, len(prices)):
        momentum[i] = prices[i] - prices[i - window]
    return momentum

# Stochastic Oscillator
@njit(cache=True)
def stochastic_oscillator(prices, window=14):
    stoch_k = np.full(len(prices), np.nan)
    for i in range(window - 1, len(prices)):
        highest_high = np.max(prices[i - window + 1:i + 1])
        lowest_low = np.min(prices[i - window + 1:i + 1])
        stoch_k[i] = ((prices[i] - lowest_low) / (highest_high - lowest_low)) * 100 if highest_high != lowest_low else 0
    return stoch_k

# Average True Range (ATR)
@njit(cache=True)
def average_true_range(high, low, close, window=14):
    atr = np.full(len(close), np.nan)
    true_range = np.zeros(len(close))
    
    for i in range(1, len(close)):
        high_low = high[i] - low[i]
        high_close = np.abs(high[i] - close[i - 1])
        low_close = np.abs(low[i] - close[i - 1])
        true_range[i] = max(high_low, high_close, low_close)
    
    atr[window - 1] = np.mean(true_range[1:window])
    for i in range(window, len(close)):
        atr[i] = (atr[i - 1] * (window - 1) + true_range[i]) / window
    
    return atr

# True Range (TR)
@njit(cache=True)
def true_range(high, low, close, window=14):

    true_range = np.zeros(len(close))
    
    for i in range(1, len(close)):
        high_low = high[i] - low[i]
        high_close = np.abs(high[i] - close[i - 1])
        low_close = np.abs(low[i] - close[i - 1])
        true_range[i] = max(high_low, high_close, low_close)
    
    return true_range

@njit(cache=True)
def moving_average_difference(prices, short_window, long_window):
    madi = np.full(len(prices), np.nan)
    
    # Calculate short and long moving averages
    short_ma = np.zeros(len(prices))
    long_ma = np.zeros(len(prices))
    
    # Initialize with -999 for periods where calculation isn't possible
    for i in range(short_window - 1, len(prices)):
        short_ma[i] = np.mean(prices[i - short_window + 1:i + 1])
    
    for i in range(long_window - 1, len(prices)):
        long_ma[i] = np.mean(prices[i - long_window + 1:i + 1])
        
    # Calculate MADI
    for i in range(len(prices)):
        if i >= long_window - 1:  # Ensure both MAs have values
            madi[i] = short_ma[i] - long_ma[i]
    
    return madi

@njit(cache=True)
def linear_perc_atr(high, low, close, window=14):
    linear_perc_atr = np.full(len(close), np.nan)
    true_range = np.zeros(len(close))
    
    # Calculate True Range
    for i in range(1, len(close)):
        high_low = high[i] - low[i]
        high_close = np.abs(high[i] - close[i - 1])
        low_close = np.abs(low[i] - close[i - 1])
        true_range[i] = max(high_low, high_close, low_close)
    
    # Calculate ATR and scale by close price to get percentage
    atr = np.zeros(len(close))
    atr[window - 1] = np.mean(true_range[1:window])
    
    for i in range(window, len(close)):
        atr[i] = (atr[i - 1] * (window - 1) + true_range[i]) / window
        linear_perc_atr[i] = (atr[i] / close[i]) * 100  # ATR as a percentage of closing price

    return linear_perc_atr

# Quadratic Percentage Average True Range (Quadratic_Per_ATR)
@njit(cache=True)
def quadratic_perc_atr(high, low, close, window=14):
    quadratic_perc_atr = np.full(len(close), np.nan)
    true_range = np.zeros(len(close))
    
    for i in range(1, len(close)):
        high_low = high[i] - low[i]
        high_close = np.abs(high[i] - close[i - 1])
        low_close = np.abs(low[i] - close[i - 1])
        true_range[i] = max(high_low, high_close, low_close)
    
    atr = np.zeros(len(close))
    atr[window - 1] = np.mean(true_range[1:window])
    
    for i in range(window, len(close)):
        atr[i] = (atr[i - 1] * (window - 1) + true_range[i]) / window
        quadratic_perc_atr[i] = ((atr[i] / close[i]) ** 2) * 100

    return quadratic_perc_atr

# Cubic Percentage Average True Range (Cubic_Per_ATR)
@njit(cache=True)
def cubic_perc_atr(high, low, close, window=14):
    cubic_perc_atr = np.full(len(close), np.nan)
    true_range = np.zeros(len(close))
    
    for i in range(1, len(close)):
        high_low = high[i] - low[i]
        high_close = np.abs(high[i] - close[i - 1])
        low_close = np.abs(low[i] - close[i - 1])
        true_range[i] = max(high_low, high_close, low_close)
    
    atr = np.zeros(len(close))
    atr[window - 1] = np.mean(true_range[1:window])
    
    for i in range(window, len(close)):
        atr[i] = (atr[i - 1] * (window - 1) + true_range[i]) / window
        cubic_perc_atr[i] = ((atr[i] / close[i]) ** 3) * 100

    return cubic_perc_atr

# Average Directional Index (ADX)
@njit(cache=True)
def adx(high, low, close, window=14):
    adx = np.full(len(close), np.nan)
    dx = np.zeros(len(close))
    
    for i in range(1, len(close)):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0
        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

        dx[i] = 100 * abs(plus_dm - minus_dm) / tr if tr > 0 else 0

    adx[window - 1] = np.mean(dx[1:window])
    
    for i in range(window, len(close)):
        adx[i] = (adx[i - 1] * (window - 1) + dx[i]) / window
    
    return adx

# Min_ADX and Max_ADX
@njit(cache=True)
def min_max_adx(adx_values, window=14):
    min_adx = np.full(len(adx_values), np.nan)
    max_adx = np.full(len(adx_values), np.nan)
    
    for i in range(window - 1, len(adx_values)):
        min_adx[i] = np.min(adx_values[i - window + 1:i + 1])
        max_adx[i] = np.max(adx_values[i - window + 1:i + 1])
    
    return min_adx, max_adx

# Residual_Min_ADX and Residual_Max_ADX
@njit(cache=True)
def residual_min_max_adx(adx_values, window=14):
    min_adx, max_adx = min_max_adx(adx_values, window)
    residual_min_adx = np.full(len(adx_values), np.nan)
    residual_max_adx = np.full(len(adx_values), np.nan)
    
    for i in range(len(adx_values)):
        if not np.isnan(min_adx[i]):
            residual_min_adx[i] = adx_values[i] - min_adx[i]
        if not np.isnan(max_adx[i]):
            residual_max_adx[i] = max_adx[i] - adx_values[i]
    
    return residual_min_adx, residual_max_adx

# Delta_ADX and Accel_ADX
@njit(cache=True)
def delta_accel_adx(adx_values, window=14):
    delta_adx = np.full(len(adx_values), np.nan)
    accel_adx = np.full(len(adx_values), np.nan)
    
    for i in range(1, len(adx_values)):
        if (not np.isnan(adx_values[i])) and (not np.isnan(adx_values[i - 1])):
            delta_adx[i] = adx_values[i] - adx_values[i - 1]
    
    for i in range(1, len(delta_adx)):
        if (not np.isnan(delta_adx[i])) and (not np.isnan(delta_adx[i - 1])):
            accel_adx[i] = delta_adx[i] - delta_adx[i - 1]
    
    return delta_adx, accel_adx

# Intraday Intensity
@njit(cache=True)
def intraday_intensity(high, low, close, volume):
    intraday_intensity = np.full(len(close), np.nan)
    
    for i in range(len(close)):
        high_low = high[i] - low[i]
        if high_low != 0:
            intraday_intensity[i] = ((close[i] - low[i]) - (high[i] - close[i])) / high_low * volume[i]
    
    return intraday_intensity

@njit(cache=True)
def delta_intraday_intensity(high, low, close, volume):
    delta_intraday_intensity = np.full(len(close), np.nan)
    intraday_intensity = np.full(len(close), np.nan)
    
    # Calculate Intraday Intensity
    for i in range(len(close)):
        high_low = high[i] - low[i]
        if high_low != 0:
            intraday_intensity[i] = ((close[i] - low[i]) - (high[i] - close[i])) / high_low * volume[i]
    
    # Calculate Delta Intraday Intensity
    for i in range(1, len(close)):
        if (not np.isnan(intraday_intensity[i])) and (not np.isnan(intraday_intensity[i - 1])):
            delta_intraday_intensity[i] = intraday_intensity[i] - intraday_intensity[i - 1]
    
    return delta_intraday_intensity

# Reactivity
@njit(cache=True)
def reactivity(prices, window=14):
    reactivity = np.full(len(prices), np.nan)
    for i in range(window, len(prices)):
        reactivity[i] = prices[i] - prices[i - window]
    return reactivity

# Delta Reactivity
@njit(cache=True)
def delta_reactivity(reactivity_values):
    delta_reactivity = np.full(len(reactivity_values), np.nan)
    for i in range(1, len(reactivity_values)):
        if (not np.isnan(reactivity_values[i])) and (not np.isnan(reactivity_values[i - 1])):
            delta_reactivity[i] = reactivity_values[i] - reactivity_values[i - 1]
    return delta_reactivity

# Min Reactivity
@njit(cache=True)
def min_reactivity(reactivity_values, window=14):
    min_reactivity = np.full(len(reactivity_values), np.nan)
    for i in range(window - 1, len(reactivity_values)):
        min_reactivity[i] = np.min(reactivity_values[i - window + 1:i + 1])
    return min_reactivity

# Max Reactivity
@njit(cache=True)
def max_reactivity(reactivity_values, window=14):
    max_reactivity = np.full(len(reactivity_values), np.nan)
    for i in range(window - 1, len(reactivity_values)):
        max_reactivity[i] = np.max(reactivity_values[i - window + 1:i + 1])
    return max_reactivity

@njit(cache=True)
def close_to_close(close):
    close_to_close_diff = np.full(len(close), np.nan)
    for i in range(1, len(close)):
        close_to_close_diff[i] = close[i] - close[i - 1]
    return close_to_close_diff

# N-Day High
@njit(cache=True)
def n_day_high(prices, n):
    n_day_high = np.full(len(prices), np.nan)
    for i in range(n - 1, len(prices)):
        n_day_high[i] = np.max(prices[i - n + 1:i + 1])
    return n_day_high

# N-Day Low
@njit(cache=True)
def n_day_low(prices, n):
    n_day_low = np.full(len(prices), np.nan)
    for i in range(n - 1, len(prices)):
        n_day_low[i] = np.min(prices[i - n + 1:i + 1])
    return n_day_low

@njit(cache=True)
def close_minus_moving_average(close, window):
    close_ma_diff = np.full(len(close), np.nan)
    moving_avg = np.zeros(len(close))
    
    for i in range(window - 1, len(close)):
        moving_avg[i] = np.mean(close[i - window + 1:i + 1])
        close_ma_diff[i] = close[i] - moving_avg[i]
    
    return close_ma_diff

# Linear Deviation
@njit(cache=True)
def linear_deviation(prices, window):
    linear_dev = np.full(len(prices), np.nan)
    for i in range(window - 1, len(prices)):
        mean_price = np.mean(prices[i - window + 1:i + 1])
        linear_dev[i] = abs(prices[i] - mean_price)
    return linear_dev

# Quadratic Deviation
@njit(cache=True)
def quadratic_deviation(prices, window):
    quadratic_dev = np.full(len(prices), np.nan)
    for i in range(window - 1, len(prices)):
        mean_price = np.mean(prices[i - window + 1:i + 1])
        quadratic_dev[i] = (prices[i] - mean_price) ** 2
    return quadratic_dev

# Cubic Deviation
@njit(cache=True)
def cubic_deviation(prices, window):
    cubic_dev = np.full(len(prices), np.nan)
    for i in range(window - 1, len(prices)):
        mean_price = np.mean(prices[i - window + 1:i + 1])
        cubic_dev[i] = abs(prices[i] - mean_price) ** 3
    return cubic_dev

@njit(cache=True)
def detrended_rsi(close, window):
    rsi_values = np.full(len(close), np.nan)
    delta = np.diff(close)
    delta = np.append(0, delta)
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    
    avg_gain[window] = np.mean(up[1:window+1])
    avg_loss[window] = np.mean(down[1:window+1])
    
    for i in range(window + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (window - 1) + up[i]) / window
        avg_loss[i] = (avg_loss[i - 1] * (window - 1) + down[i]) / window
        
        rs = avg_gain[i] / avg_loss[i] if avg_loss[i] > 0 else 0
        rsi_values[i] = 100 - (100 / (1 + rs))
    
    ma_rsi = close_minus_moving_average(rsi_values, window)
    
    detrended_rsi_values = np.full(len(close), np.nan)
    for i in range(len(close)):
        if (not np.isnan(rsi_values[i])) and (not np.isnan(ma_rsi[i])):
            detrended_rsi_values[i] = rsi_values[i] - ma_rsi[i]
    
    return detrended_rsi_values

@njit(cache=True)
def abs_price_change_oscillator(close, window):
    abs_change = np.full(len(close), np.nan)
    for i in range(window, len(close)):
        abs_change[i] = abs(close[i] - close[i - window])
    return abs_change

@njit(cache=True)
def atr_ratio(high, low, close, window=14):
    atr = np.full(len(close), np.nan)
    true_range = np.zeros(len(close))
    
    for i in range(1, len(close)):
        high_low = high[i] - low[i]
        high_close = np.abs(high[i] - close[i - 1])
        low_close = np.abs(low[i] - close[i - 1])
        true_range[i] = max(high_low, high_close, low_close)
    
    atr[window - 1] = np.mean(true_range[1:window])
    for i in range(window, len(close)):
        atr[i] = (atr[i - 1] * (window - 1) + true_range[i]) / window
    
    atr_ratio = np.full(len(close), np.nan)
    for i in range(len(close)):
        if not np.isnan(atr[i]):
            atr_ratio[i] = atr[i] / close[i]
    
    return atr_ratio

@njit(cache=True)
def n_day_narrower_wider(high, low, n):
    narrower = np.full(len(high), np.nan)
    wider = np.full(len(high), np.nan)
    
    for i in range(n - 1, len(high)):
        avg_range = np.mean(high[i - n + 1:i + 1] - low[i - n + 1:i + 1])
        current_range = high[i] - low[i]
        narrower[i] = 1 if current_range < avg_range else 0
        wider[i] = 1 if current_range > avg_range else 0
    
    return narrower, wider

# Price Skewness
@njit(cache=True)
def price_skewness(prices, window):
    skewness = np.full(len(prices), np.nan)
    
    for i in range(window - 1, len(prices)):
        window_data = prices[i - window + 1:i + 1]
        mean = np.mean(window_data)
        std_dev = np.std(window_data)
        
        if std_dev > 0:
            skewness[i] = np.mean(((window_data - mean) / std_dev) ** 3)
    
    return skewness

# Change Skewness
@njit(cache=True)
def change_skewness(prices, window):
    change_skewness = np.full(len(prices), np.nan)
    price_changes = np.diff(prices)
    price_changes = np.append(0, price_changes)
    
    for i in range(window - 1, len(prices)):
        window_changes = price_changes[i - window + 1:i + 1]
        mean = np.mean(window_changes)
        std_dev = np.std(window_changes)
        
        if std_dev > 0:
            change_skewness[i] = np.mean(((window_changes - mean) / std_dev) ** 3)
    
    return change_skewness

# Price Kurtosis
@njit(cache=True)
def price_kurtosis(prices, window):
    kurtosis = np.full(len(prices), np.nan)
    
    for i in range(window - 1, len(prices)):
        window_data = prices[i - window + 1:i + 1]
        mean = np.mean(window_data)
        std_dev = np.std(window_data)
        
        if std_dev > 0:
            kurtosis[i] = np.mean(((window_data - mean) / std_dev) ** 4) - 3
    
    return kurtosis

# Change Kurtosis
@njit(cache=True)
def change_kurtosis(prices, window):
    change_kurtosis = np.full(len(prices), np.nan)
    price_changes = np.diff(prices)
    price_changes = np.append(0, price_changes)
    
    for i in range(window - 1, len(prices)):
        window_changes = price_changes[i - window + 1:i + 1]
        mean = np.mean(window_changes)
        std_dev = np.std(window_changes)
        
        if std_dev > 0:
            change_kurtosis[i] = np.mean(((window_changes - mean) / std_dev) ** 4) - 3
    
    return change_kurtosis

# Delta Price Skewness
@njit(cache=True)
def delta_price_skewness(price_skewness_values):
    delta_skewness = np.full(len(price_skewness_values), np.nan)
    
    for i in range(1, len(price_skewness_values)):
        if (not np.isnan(price_skewness_values[i])) and (not np.isnan(price_skewness_values[i - 1])):
            delta_skewness[i] = price_skewness_values[i] - price_skewness_values[i - 1]
    
    return delta_skewness

# Delta Change Skewness
@njit(cache=True)
def delta_change_skewness(change_skewness_values):
    delta_skewness = np.full(len(change_skewness_values), np.nan)
    
    for i in range(1, len(change_skewness_values)):
        if (not np.isnan(change_skewness_values[i])) and (not np.isnan(change_skewness_values[i - 1])):
            delta_skewness[i] = change_skewness_values[i] - change_skewness_values[i - 1]
    
    return delta_skewness

# Delta Price Kurtosis
@njit(cache=True)
def delta_price_kurtosis(price_kurtosis_values):
    delta_kurtosis = np.full(len(price_kurtosis_values), np.nan)
    
    for i in range(1, len(price_kurtosis_values)):
        if (not np.isnan(price_kurtosis_values[i])) and (not np.isnan(price_kurtosis_values[i - 1])):
            delta_kurtosis[i] = price_kurtosis_values[i] - price_kurtosis_values[i - 1]
    
    return delta_kurtosis

# Delta Change Kurtosis
@njit(cache=True)
def delta_change_kurtosis(change_kurtosis_values):
    delta_kurtosis = np.full(len(change_kurtosis_values), np.nan)
    
    for i in range(1, len(change_kurtosis_values)):
        if (not np.isnan(change_kurtosis_values[i])) and (not np.isnan(change_kurtosis_values[i - 1])):
            delta_kurtosis[i] = change_kurtosis_values[i] - change_kurtosis_values[i - 1]
    
    return delta_kurtosis

# Volume Momentum
@njit(cache=True)
def volume_momentum(volume, window):
    momentum = np.full(len(volume), np.nan)
    
    for i in range(window - 1, len(volume)):
        momentum[i] = volume[i] - volume[i - window]
    
    return momentum

# Delta Volume Momentum
@njit(cache=True)
def delta_volume_momentum(volume_momentum_values):
    delta_momentum = np.full(len(volume_momentum_values), np.nan)
    
    for i in range(1, len(volume_momentum_values)):
        if (not np.isnan(volume_momentum_values[i])) and (not np.isnan(volume_momentum_values[i - 1])):
            delta_momentum[i] = volume_momentum_values[i] - volume_momentum_values[i - 1]
    
    return delta_momentum

@njit(cache=True)
def volume_weighted_ma_over_ma(price, volume, window):
    vwma = np.full(len(price), np.nan)
    sma = np.full(len(price), np.nan)
    
    for i in range(window - 1, len(price)):
        total_volume = np.sum(volume[i - window + 1:i + 1])
        vwma[i] = np.sum(price[i - window + 1:i + 1] * volume[i - window + 1:i + 1]) / total_volume if total_volume > 0 else 0
        
        sma[i] = np.mean(price[i - window + 1:i + 1])
    
    return vwma, sma

@njit(cache=True)
def diff_volume_weighted_ma_over_ma(price, volume, window):
    vwma, sma = volume_weighted_ma_over_ma(price, volume, window)
    diff_vwma_sma = np.full(len(price), np.nan)
    
    for i in range(len(price)):
        if (not np.isnan(vwma[i])) and (not np.isnan(sma[i])):
            diff_vwma_sma[i] = vwma[i] - sma[i]
    
    return diff_vwma_sma

@njit(cache=True)
def on_balance_volume(price, volume):
    obv = np.full(len(price), np.nan)
    obv[0] = volume[0] if price[0] > 0 else -volume[0]  # Start with the first value
    
    for i in range(1, len(price)):
        if price[i] > price[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif price[i] < price[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    
    return obv

@njit(cache=True)
def delta_on_balance_volume(obv_values):
    delta_obv = np.full(len(obv_values), np.nan)
    
    for i in range(1, len(obv_values)):
        if (not np.isnan(obv_values[i])) and (not np.isnan(obv_values[i - 1])):
            delta_obv[i] = obv_values[i] - obv_values[i - 1]
    
    return delta_obv

@njit(cache=True)
def positive_volume_indicator(volume):
    pvi = np.full(len(volume), np.nan)
    pvi[0] = volume[0]
    
    for i in range(1, len(volume)):
        if volume[i] > volume[i - 1]:
            pvi[i] = pvi[i - 1] + volume[i]
        else:
            pvi[i] = pvi[i - 1]
    
    return pvi

@njit(cache=True)
def negative_volume_indicator(volume):
    nvi = np.full(len(volume), np.nan)
    nvi[0] = volume[0]
    
    for i in range(1, len(volume)):
        if volume[i] < volume[i - 1]:
            nvi[i] = nvi[i - 1] + volume[i]
        else:
            nvi[i] = nvi[i - 1]
    
    return nvi

@njit(cache=True)
def product_price_volume(price, volume):
    product = np.full(len(price), np.nan)
    
    for i in range(len(price)):
        product[i] = price[i] * volume[i]
    
    return product

@njit(cache=True)
def sum_price_volume(price, volume, window):
    sum_pv = np.full(len(price), np.nan)
    
    for i in range(window - 1, len(price)):
        sum_pv[i] = np.sum(price[i - window + 1:i + 1] * volume[i - window + 1:i + 1])
    
    return sum_pv

@njit(cache=True)
def custom_histogram(data, bins):
    hist = np.zeros(bins, dtype=np.int32)
    min_val = np.min(data)
    max_val = np.max(data)
    bin_width = (max_val - min_val) / bins
    
    for value in data:
        if value < min_val or value >= max_val:
            continue
        index = int((value - min_val) / bin_width)
        if index == bins:  # Make sure the index is within bounds
            index -= 1
        hist[index] += 1

    return hist

@njit(cache=True)
def price_entropy(prices, window, bins=10):
    entropy = np.full(len(prices), np.nan)

    for i in range(window - 1, len(prices)):
        hist = custom_histogram(prices[i - window + 1:i + 1], bins)
        hist = hist[hist > 0]  # Filter out zero entries
        if hist.size > 0:
            prob = hist / np.sum(hist)
            entropy[i] = -np.sum(prob * np.log(prob))  # Calculate entropy

    return entropy

@njit(cache=True)
def volume_entropy(volume, window, bins=10):
    entropy = np.full(len(volume), np.nan)

    for i in range(window - 1, len(volume)):
        hist = custom_histogram(volume[i - window + 1:i + 1], bins)
        hist = hist[hist > 0]  # Filter out zero entries
        if hist.size > 0:
            prob = hist / np.sum(hist)
            entropy[i] = -np.sum(prob * np.log(prob))  # Calculate entropy

    return entropy

@njit(cache=True)
def calculate_moving_support_resistance(prices, window, lag):
    n = len(prices)
    support = np.full(n, np.nan)
    resistance = np.full(n, np.nan)

    for i in range(window - 1, n):
        # Calculate the highest high and lowest low over the window
        high = np.max(prices[i - window + 1:i + 1])
        low = np.min(prices[i - window + 1:i + 1])

        # Lag the support and resistance levels
        if i >= lag:
            resistance[i] = high if high > resistance[i - lag] else resistance[i - lag]
            support[i] = low if low < support[i - lag] else support[i - lag]
        else:
            resistance[i] = high
            support[i] = low

    return support, resistance

@njit(cache=True)
def chaikin_ad_line(close, low, high, volume):
    n = len(close)
    ad_line = np.full(n, np.nan)
    for i in range(1, n):
        money_flow_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i]) if (high[i] - low[i]) != 0 else 0
        ad_line[i] = ad_line[i - 1] + money_flow_multiplier * volume[i]
    return ad_line

@njit(cache=True)
def chaikin_ad_oscillator(ad_line, short_window, long_window):
    n = len(ad_line)
    oscillator = np.full(n, np.nan)
    short_ma = np.full(n, np.nan)
    long_ma = np.full(n, np.nan)

    for i in range(short_window - 1, n):
        short_ma[i] = np.mean(ad_line[i - short_window + 1:i + 1])
    for i in range(long_window - 1, n):
        long_ma[i] = np.mean(ad_line[i - long_window + 1:i + 1])
    
    for i in range(max(short_window, long_window) - 1, n):
        if (not np.isnan(long_ma[i])):
            oscillator[i] = short_ma[i] - long_ma[i]
    return oscillator

@njit(cache=True)
def absolute_price_oscillator(close, short_window, long_window):
    n = len(close)
    apo = np.full(n, np.nan)
    short_ma = np.full(n, np.nan)
    long_ma = np.full(n, np.nan)

    for i in range(short_window - 1, n):
        short_ma[i] = np.mean(close[i - short_window + 1:i + 1])
    for i in range(long_window - 1, n):
        long_ma[i] = np.mean(close[i - long_window + 1:i + 1])

    for i in range(max(short_window, long_window) - 1, n):
        if not np.isnan(long_ma[i]):
            apo[i] = short_ma[i] - long_ma[i]
    return apo

@njit(cache=True)
def aroon(prices, period):
    n = len(prices)
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)

    for i in range(period - 1, n):
        highest_high = np.argmax(prices[i - period + 1:i + 1]) + (i - period + 1)
        lowest_low = np.argmin(prices[i - period + 1:i + 1]) + (i - period + 1)

        aroon_up[i] = (period - (i - highest_high)) / period * 100
        aroon_down[i] = (period - (i - lowest_low)) / period * 100

    return aroon_up, aroon_down

@njit(cache=True)
def balance_of_power(close, high, low):
    n = len(close)
    bop = np.full(n, np.nan)
    
    for i in range(1, n):
        bop[i] = (close[i] - low[i]) - (high[i] - close[i])
    
    return bop

@njit(cache=True)
def double_exponential_moving_average(close, period):
    n = len(close)
    dema = np.full(n, np.nan)
    ema = np.full(n, np.nan)

    # Calculate the EMA
    for i in range(period - 1, n):
        ema[i] = np.mean(close[i - period + 1:i + 1])

    # Calculate DEMA
    for i in range(period - 1, n):
        ema_dema = np.mean(ema[i - period + 1:i + 1])
        dema[i] = 2 * ema[i] - ema_dema
    
    return dema

@njit(cache=True)
def directional_movement_index(high, low, close, period):
    n = len(close)
    dmi_plus = np.full(n, np.nan)
    dmi_minus = np.full(n, np.nan)
    dx = np.full(n, np.nan)

    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0

        dmi_plus[i] = plus_dm
        dmi_minus[i] = minus_dm
    
    # Calculate the DMI
    for i in range(period - 1, n):
        avg_plus_dm = np.mean(dmi_plus[i - period + 1:i + 1])
        avg_minus_dm = np.mean(dmi_minus[i - period + 1:i + 1])
        if avg_plus_dm + avg_minus_dm != 0:
            dx[i] = (avg_plus_dm - avg_minus_dm) / (avg_plus_dm + avg_minus_dm) * 100
    
    return dmi_plus, dmi_minus, dx

@njit(cache=True)
def hilbert_dominant_cycle_period(close, period):
    n = len(close)
    cycle_period = np.full(n, np.nan)

    for i in range(period - 1, n):
        cycle_period[i] = np.mean(close[i - period + 1:i + 1])  # Placeholder logic for dominant cycle period

    return cycle_period

@njit(cache=True)
def hilbert_dominant_cycle_phase(close, period):
    n = len(close)
    cycle_phase = np.full(n, np.nan)

    for i in range(period - 1, n):
        cycle_phase[i] = np.mean(close[i - period + 1:i + 1])  # Placeholder logic for dominant cycle phase

    return cycle_phase

@njit(cache=True)
def hilbert_phasor_components(close):
    n = len(close)
    phasor_components = np.full((n, 2), np.nan)  # [cos, sin]

    for i in range(1, n):
        phasor_components[i, 0] = np.cos(close[i])  # Placeholder
        phasor_components[i, 1] = np.sin(close[i])  # Placeholder

    return phasor_components

@njit(cache=True)
def hilbert_sinewave(close):
    n = len(close)
    sine_wave = np.full(n, np.nan)

    for i in range(1, n):
        sine_wave[i] = np.sin(close[i])  # Placeholder logic for sine wave

    return sine_wave

@njit(cache=True)
def hilbert_instantaneous_trendline(close):
    n = len(close)
    trendline = np.full(n, np.nan)

    for i in range(1, n):
        trendline[i] = close[i]  # Placeholder logic for trendline

    return trendline

@njit(cache=True)
def hilbert_trend_vs_cycle_mode(close):
    n = len(close)
    trend_vs_cycle = np.full(n, np.nan)

    for i in range(1, n):
        trend_vs_cycle[i] = close[i]  # Placeholder logic for trend vs cycle

    return trend_vs_cycle

@njit(cache=True)
def mesa_adaptive_moving_average(close, period):
    n = len(close)
    mesa_ama = np.full(n, np.nan)

    for i in range(period, n):
        mesa_ama[i] = np.mean(close[i - period + 1:i + 1])  # Placeholder logic for MESA

    return mesa_ama

@njit(cache=True)
def median_price(high, low):
    n = len(high)
    median_prices = np.full(n, np.nan)

    for i in range(n):
        median_prices[i] = (high[i] + low[i]) / 2.0
    
    return median_prices

@njit(cache=True)
def midpoint_over_period(close, period):
    n = len(close)
    midpoints = np.full(n, np.nan)

    for i in range(period - 1, n):
        midpoints[i] = (max(close[i - period + 1:i + 1]) + min(close[i - period + 1:i + 1])) / 2.0

    return midpoints

@njit(cache=True)
def parabolic_sar(high, low, initial_af=0.02, max_af=0.2):
    n = len(high)
    sar = np.full(n, np.nan)
    up_trend = True
    ep = high[0]
    af = initial_af

    for i in range(1, n):
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

        if up_trend:
            if high[i] > ep:
                ep = high[i]
                af = min(af + initial_af, max_af)
            if low[i] < sar[i]:
                up_trend = False
                sar[i] = ep
                ep = low[i]
                af = initial_af
        else:
            if low[i] < ep:
                ep = low[i]
                af = min(af + initial_af, max_af)
            if high[i] > sar[i]:
                up_trend = True
                sar[i] = ep
                ep = high[i]
                af = initial_af

    return sar

@njit(cache=True)
def triple_exponential_moving_average(close, period):
    n = len(close)
    tema = np.full(n, np.nan)
    ema1 = np.full(n, np.nan)
    ema2 = np.full(n, np.nan)

    # First EMA
    for i in range(period - 1, n):
        ema1[i] = np.mean(close[i - period + 1:i + 1])

    # Second EMA
    for i in range(period - 1, n):
        ema2[i] = np.mean(ema1[i - period + 1:i + 1])

    # Third EMA
    for i in range(period - 1, n):
        tema[i] = 3 * ema1[i] - 3 * ema2[i] + np.mean(ema2[i - period + 1:i + 1])

    return tema

@njit(cache=True)
def williams_r(high, low, close, period):
    n = len(close)
    wr = np.full(n, np.nan)

    for i in range(period - 1, n):
        highest_high = max(high[i - period + 1:i + 1])
        lowest_low = min(low[i - period + 1:i + 1])
        wr[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100 if (highest_high - lowest_low) != 0 else -999

    return wr

@njit(cache=True)
def z_score(array):
    mean = np.mean(array)
    std_dev = np.std(array)
    return (array - mean) / std_dev

@njit(cache=True)
def rolling_drawdown(data, window_size):
    n = len(data)
    drawdowns = np.full(n, np.nan)
    
    for i in range(n - window_size + 1):
        window = data[i:i + window_size]
        peak = np.max(window)
        current_value = window[-1]
        drawdowns[i + window_size - 1] = (peak - current_value) / peak if peak != 0 else 0.0

    return drawdowns

@njit(cache=True)
def rolling_volatility(data, window_size):
    n = len(data)
    volatilities = np.full(n, np.nan)
    
    for i in range(n - window_size + 1):
        window = data[i:i + window_size]
        volatilities[i + window_size - 1] = np.std(window)
    
    return volatilities

@njit(cache=True)
def rolling_parkinson_estimator(high, low, window_size):
    n = len(high)
    rolling_volatility = np.full(n, np.nan)
    
    for i in range(window_size - 1, n):
        window_high = high[i - window_size + 1:i + 1]
        window_low = low[i - window_size + 1:i + 1]
        
        N = len(window_high)
        sum_squared = np.sum(np.log(window_high / window_low) ** 2)
        
        if N > 0:
            volatility = math.sqrt((1 / (4 * N * math.log(2))) * sum_squared)
            rolling_volatility[i] = volatility

    return rolling_volatility

@njit(cache=True)
def rolling_rogers_satchell_estimator(open, high, low, close, window_size):
    n = len(open)
    rolling_volatility = np.full(n, np.nan)
    
    for i in range(window_size - 1, n):
        window_open = open[i - window_size + 1:i + 1]
        window_high = high[i - window_size + 1:i + 1]
        window_low = low[i - window_size + 1:i + 1]
        window_close = close[i - window_size + 1:i + 1]

        sum_squared = 0
        is_continue = 0

        for j in range(len(window_open)):

            if np.isnan(window_open[j]) or np.isnan(window_high[j]) or np.isnan(window_low[j]) or np.isnan(window_close[j]):
                is_continue = 1
                break
            else:
                sum_squared += np.log(window_high[j] / window_open[j]) * np.log(window_high[j] / window_close[j]) + \
                    np.log(window_low[j] / window_open[j]) * np.log(window_low[j] / window_close[j])

        if is_continue == 1:
            continue     

        N = len(window_high)
        
        if N > 0:
            volatility = math.sqrt(sum_squared / N)
            rolling_volatility[i] = volatility

    return rolling_volatility

@njit(cache=True)
def rolling_yang_zhang_estimator(open, high, low, close, window_size):

    n = len(open)
    
    rolling_volatility = np.full(n, np.nan)
    
    k = 0.34 / (1.34 + (window_size + 1) / (window_size - 1))
    
    for i in range(window_size, n):
        window_open = open[i - window_size + 1:i + 1]
        window_high = high[i - window_size + 1:i + 1]
        window_low = low[i - window_size + 1:i + 1]
        window_close = close[i - window_size + 1:i + 1]
        window_open_prev = open[i - window_size:i]
        window_close_prev = close[i - window_size:i]

        N = len(window_high)

        is_continue = 0
        oc_var = 0
        co_var = 0
        rs_var = 0

        for j in range(len(window_open)):
            if np.isnan(window_open[j]) or np.isnan(window_open_prev[j]):
                is_continue = 1
                break
            else:
                oc_var += (np.log(window_close[j] / window_open[j]) ** 2) / (N - 1)
                co_var += (np.log(window_open[j] / window_close_prev[j]) ** 2) / (N - 1)
                rs_var += (np.log(window_high[j] / window_open[j]) * np.log(window_high[j] / window_close[j]) +
                    np.log(window_low[j] / window_open[j]) * np.log(window_low[j] / window_close[j])) / (N - 1)
                
        if is_continue == 1:
            continue
        
        volatility = math.sqrt(co_var + k * oc_var + (1 - k) * rs_var)
        rolling_volatility[i] = volatility

    return rolling_volatility

@njit(cache=True)
def rolling_garman_klass_estimator(open, high, low, close, window_size):

    n = len(open)

    rolling_volatility = np.full(n, np.nan)
    
    for i in range(window_size - 1, n):

        window_open = open[i - window_size + 1:i + 1]
        window_high = high[i - window_size + 1:i + 1]
        window_low = low[i - window_size + 1:i + 1]
        window_close = close[i - window_size + 1:i + 1]

        is_continue = 0
        sum_squared = 0

        for j in range(len(window_open)):
            if np.isnan(window_open[j]):
                is_continue = 1
                break
            else:
                sum_squared += (0.5 * (np.log(window_high[j] / window_low[j]) ** 2) - 
                    (2 * np.log(2) - 1) * (np.log(window_close[j] / window_open[j]) ** 2))

        if is_continue == 1:
            continue

        N = len(window_high)
        
        if N > 0:
            volatility = math.sqrt(sum_squared / N)
            if not np.isnan(volatility):
                rolling_volatility[i] = volatility

    return rolling_volatility

@njit(cache=True)
def ultimate_oscillator(high, low, close, period1, period2, period3):
    n = len(close)
    uo = np.full(n, np.nan)
    
    if n < period3:
        return uo
    
    bp = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        bp[i] = close[i] - min(low[i], close[i-1])
        tr[i] = max(high[i], close[i-1]) - min(low[i], close[i-1])
    
    def sma(arr, period):
        result = np.zeros(n)
        cumsum = 0.0
        for i in range(period):
            cumsum += arr[i]
        result[period - 1] = cumsum / period
        for i in range(period, n):
            cumsum = cumsum - arr[i - period] + arr[i]
            result[i] = cumsum / period
        return result
    
    avgBP1 = sma(bp, period1)
    avgTR1 = sma(tr, period1)
    avgBP2 = sma(bp, period2)
    avgTR2 = sma(tr, period2)
    avgBP3 = sma(bp, period3)
    avgTR3 = sma(tr, period3)
    
    for i in range(period3 - 1, n):
        weighted_bp = 4 * avgBP1[i] + 2 * avgBP2[i] + avgBP3[i]
        weighted_tr = 4 * avgTR1[i] + 2 * avgTR2[i] + avgTR3[i]
        if weighted_tr != 0:
            uo[i] = 100 * (weighted_bp / weighted_tr)
    
    return uo

@njit(cache=True)
def medprice(high, low):
    return (high + low) / 2

@njit(cache=True)
def ldecay(price, period):
    
    decay_values = np.full_like(price, np.nan)
    weights = np.arange(1, period + 1)
    weights = weights / weights.sum()
    
    for i in range(period - 1, len(price)):
        decay_values[i] = np.dot(price[i-period+1:i+1], weights)
    
    return decay_values

@njit(cache=True)
def logret(price):
    
    log_returns = np.full_like(price, np.nan)
    
    for i in range(1, len(price)):
        log_returns[i] = np.log(price[i] / price[i - 1])
    
    return log_returns

@njit(cache=True)
def pvi(price, volume):

    pvi_values = np.full_like(price, np.nan)
    pvi_values[0] = 1000
    
    for i in range(1, len(price)):
        if volume[i] > volume[i - 1]:
            pvi_values[i] = pvi_values[i - 1] + (price[i] - price[i - 1]) / price[i - 1] * pvi_values[i - 1]
        else:
            pvi_values[i] = pvi_values[i - 1]
    
    return pvi_values

@njit(cache=True)
def pctret(price):
    
    pct_returns = np.full_like(price, np.nan)
    
    for i in range(1, len(price)):
        pct_returns[i] = (price[i] - price[i - 1]) / price[i - 1] * 100
    
    return pct_returns

@njit(cache=True)
def cti(price, period):
    
    cti_values = np.full_like(price, np.nan)
    
    for i in range(period, len(price)):
        current_price = price[i]
        high = np.max(price[i - period + 1:i + 1])
        low = np.min(price[i - period + 1:i + 1])
        cti_values[i] = (2 * (current_price - low) / (high - low) - 1) * 100 if high != low else 0
    
    return cti_values

@njit(cache=True)
def dema(price, period):
    
    ema1 = np.full_like(price, np.nan)
    ema2 = np.full_like(price, np.nan)
    dema_values = np.full_like(price, np.nan)
    alpha = 2 / (period + 1)
    
    # Calculate first EMA
    ema1[period - 1] = np.mean(price[:period])  # Initial value for the first EMA
    for i in range(period, len(price)):
        ema1[i] = alpha * price[i] + (1 - alpha) * ema1[i - 1]
    
    # Calculate second EMA on the first EMA
    ema2[period - 1] = ema1[period - 1]
    for i in range(period, len(ema1)):
        ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i - 1]
    
    # DEMA is calculated as: 2 * EMA1 - EMA2
    for i in range(period - 1, len(price)):
        dema_values[i] = 2 * ema1[i] - ema2[i]
    
    return dema_values

@njit(cache=True)
def hma(price, period):
    
    half_length = period // 2
    sqrt_length = int(np.sqrt(period))
    
    wma_half = np.full_like(price, np.nan)
    wma_full = np.full_like(price, np.nan)
    hma_values = np.full_like(price, np.nan)
    
    # Create weights arrays as float64 for compatibility with Numba
    half_weights = np.arange(1, half_length + 1, dtype=np.float64)
    full_weights = np.arange(1, period + 1, dtype=np.float64)
    sqrt_weights = np.arange(1, sqrt_length + 1, dtype=np.float64)
    
    # Calculate WMA for half and full period lengths
    for i in range(period - 1, len(price)):
        wma_half[i] = np.dot(price[i-half_length+1:i+1], half_weights) / np.sum(half_weights)
        wma_full[i] = np.dot(price[i-period+1:i+1], full_weights) / np.sum(full_weights)
        
    # Calculate the HMA over the square root length
    for i in range(period - 1, len(price)):
        hma_values[i] = np.dot(2 * wma_half[i-sqrt_length+1:i+1] - wma_full[i-sqrt_length+1:i+1], sqrt_weights) / np.sum(sqrt_weights)
    
    return hma_values

@njit(cache=True)
def stochastic_oscillator_kd(close_prices, high_prices, low_prices, period=14, smooth_period=3):
    n = len(close_prices)
    stoch_k = np.full(n, np.nan)  # Initialize %K line with -999
    stoch_d = np.full(n, np.nan)  # Initialize %D line with -999

    # Calculate %K values for each point in the series where there are enough values
    for i in range(period - 1, n):
        highest_high = np.max(high_prices[i - period + 1:i + 1])
        lowest_low = np.min(low_prices[i - period + 1:i + 1])
        current_close = close_prices[i]

        if highest_high != lowest_low:
            stoch_k[i] = 100 * (current_close - lowest_low) / (highest_high - lowest_low)

    # Calculate %D values as a moving average of %K, starting from smooth_period - 1 elements after %K calculation begins
    for i in range(period - 1 + smooth_period - 1, n):
        if not np.isnan(stoch_k[i - smooth_period + 1:i + 1].min()):
            stoch_d[i] = np.mean(stoch_k[i - smooth_period + 1:i + 1])

    return stoch_k, stoch_d

@njit(cache=True)
def money_flow_index(high_prices, low_prices, close_prices, volumes, period=14):
    
    n = len(close_prices)
    mfi = np.full(n, np.nan)  # Initialize MFI output array with -999

    # Calculate Typical Price and Raw Money Flow
    typical_price = (high_prices + low_prices + close_prices) / 3
    raw_money_flow = typical_price * volumes

    # Loop through each period to calculate MFI
    for i in range(period, n):
        pos_flow = 0.0  # Positive money flow
        neg_flow = 0.0  # Negative money flow
        
        # Calculate positive and negative money flow over the look-back period
        for j in range(i - period + 1, i + 1):
            if typical_price[j] > typical_price[j - 1]:
                pos_flow += raw_money_flow[j]
            elif typical_price[j] < typical_price[j - 1]:
                neg_flow += raw_money_flow[j]

        # Avoid division by zero
        if neg_flow == 0:
            mfi[i] = 100.0
        else:
            money_flow_ratio = pos_flow / neg_flow
            mfi[i] = 100 - (100 / (1 + money_flow_ratio))

    return mfi

@njit(cache=True)
def calculate_keltner_channels(high_prices, low_prices, close_prices, period=20, multiplier=2):

    n = len(close_prices)
    middle_line = np.full(n, np.nan)  # EMA of close prices
    upper_band = np.full(n, np.nan)   # Upper Keltner Channel
    lower_band = np.full(n, np.nan)   # Lower Keltner Channel
    atr_values = np.full(n, np.nan)   # ATR values
    
    # Calculate the True Range (TR) for each period
    tr = np.empty(n)
    tr[0] = 0  # First TR is undefined, can be set to zero or left out of calculation
    for i in range(1, n):
        tr[i] = max(
            high_prices[i] - low_prices[i], abs(high_prices[i] - close_prices[i - 1]), 
            abs(low_prices[i] - close_prices[i - 1])
        )

    # Calculate the Average True Range (ATR) using a simple moving average
    for i in range(period, n):
        atr_values[i] = np.mean(tr[i - period + 1:i + 1])

    # Calculate EMA of the close prices for the middle line
    alpha = 2 / (period + 1)  # EMA smoothing factor
    ema = close_prices[0]  # Start EMA with the first close price
    for i in range(n):
        if i >= period - 1:
            ema = alpha * close_prices[i] + (1 - alpha) * ema
            middle_line[i] = ema

    # Calculate Upper and Lower Keltner Channels
    for i in range(period, n):
        if not np.isnan(atr_values[i]):
            upper_band[i] = middle_line[i] + multiplier * atr_values[i]
            lower_band[i] = middle_line[i] - multiplier * atr_values[i]

    return middle_line, upper_band, lower_band

@njit(cache=True)
def commodity_channel_index(high_prices, low_prices, close_prices, period=20):
    n = len(close_prices)
    cci = np.full(n, np.nan)  # Initialize CCI output with -999
    
    # Calculate Typical Price (TP)
    typical_price = (high_prices + low_prices + close_prices) / 3
    
    # Loop through each period to calculate CCI
    for i in range(period - 1, n):
        # Calculate the Simple Moving Average (SMA) of TP
        sma_tp = np.mean(typical_price[i - period + 1:i + 1])
        
        # Calculate the Mean Deviation (MD) of TP from its SMA
        mean_dev = np.mean(np.abs(typical_price[i - period + 1:i + 1] - sma_tp))
        
        # Avoid division by zero and calculate CCI
        if mean_dev != 0:
            cci[i] = (typical_price[i] - sma_tp) / (0.015 * mean_dev)

    return cci

@njit(cache=True)
def bull_bar_tail_rolling(close_prices, open_prices, high_prices, low_prices, period=20):

    n = len(close_prices)
    bull_bar_tail_count = np.full(n, np.nan, dtype=np.int32)  # Initialize output with -999 for insufficient data
    
    # Loop through each bar to calculate the rolling count of bars meeting the conditions
    for i in range(period - 1, n):
        count = 0
        
        # Check each bar in the rolling window
        for j in range(i - period + 1, i + 1):
            # Check the conditions for a BullBarTail
            if (close_prices[j] > open_prices[j] and
                (open_prices[j] - low_prices[j]) > (close_prices[j] - open_prices[j]) and
                high_prices[j] > high_prices[j - 1]):
                count += 1
        
        # Set the count in the result array
        bull_bar_tail_count[i] = count

    return bull_bar_tail_count

@njit(cache=True)
def bear_bar_tail_rolling(close_prices, open_prices, high_prices, low_prices, period=20):
    n = len(close_prices)
    bear_bar_tail_count = np.full(n, np.nan, dtype=np.int32)  # Initialize output with -999 for insufficient data
    
    # Loop through each bar to calculate the rolling count of bars meeting the conditions
    for i in range(period - 1, n):
        count = 0
        
        # Check each bar in the rolling window
        for j in range(i - period + 1, i + 1):
            # Check the conditions for a BearBarTail
            if (close_prices[j] < open_prices[j] and
                (high_prices[j] - open_prices[j]) > (open_prices[j] - close_prices[j]) and
                low_prices[j] < low_prices[j - 1]):
                count += 1
        
        # Set the count in the result array
        bear_bar_tail_count[i] = count

    return bear_bar_tail_count

@njit(cache=True)
def awesome_oscillator(high_prices, low_prices, short_period=5, long_period=34):
    n = len(high_prices)
    ao_values = np.full(n, np.nan)  # Initialize AO array with -999 for insufficient data
    
    # Calculate the Median Price
    median_prices = (high_prices + low_prices) / 2
    
    # Calculate AO by finding the difference between the short and long period SMAs of the median price
    for i in range(long_period - 1, n):
        short_sma = np.mean(median_prices[i - short_period + 1:i + 1])
        long_sma = np.mean(median_prices[i - long_period + 1:i + 1])
        ao_values[i] = short_sma - long_sma

    return ao_values

@njit(cache=True)
def rolling_max_index(arr, window):
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.int32)  # Initialize result array with -999

    for i in range(n):
        # Check if there are enough data points for the window
        if i >= window - 1:
            # Extract the window of values
            window_values = arr[i - window + 1: i + 1]
            # Find the index of the maximum within the window
            max_index_within_window = np.argmax(window_values)
            # Convert to original array index
            result[i] = i - window + 1 + max_index_within_window

    return result

@njit(cache=True)
def rolling_min_index(arr, window):
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.int32)  # Initialize result array with -999

    for i in range(n):
        # Check if there are enough data points for the window
        if i >= window - 1:
            # Extract the window of values
            window_values = arr[i - window + 1: i + 1]
            # Find the index of the minimum within the window
            min_index_within_window = np.argmin(window_values)
            # Convert to original array index
            result[i] = i - window + 1 + min_index_within_window

    return result