import re
import numpy as np
import tqdm as tqdm
import pandas as pd
import os
from pathlib import Path
import itertools
from tqdm import tqdm
import gc

entry_txt = '''(price_data['btc_close'] < get_lag(price_data['btc_close'], lag=<lag-steps>))
(price_data['btc_close'] > get_lag(price_data['btc_close'], lag=<lag-steps>))
((price_data['day_of_week'] == 5) & (price_data['btc_close'] == signals.moving_min(price_data['btc_close'], window=<window-const>)))
((price_data['day_of_week'] == 5) & (price_data['btc_close'] == signals.moving_max(price_data['btc_close'], window=<window-const>)))
((numba_indicators_nan.moving_average(prices=price_data['btc_close'], window=<short-window-const>) < numba_indicators_nan.moving_average(prices=price_data['btc_close'], window=<long-window-const>)) & (price_data['btc_close'] < numba_indicators_nan.moving_average(prices=price_data['btc_close'], window=<short-window-const>)))
((numba_indicators_nan.moving_average(prices=price_data['btc_close'], window=<short-window-const>) > numba_indicators_nan.moving_average(prices=price_data['btc_close'], window=<long-window-const>)) & (price_data['btc_close'] > numba_indicators_nan.moving_average(prices=price_data['btc_close'], window=<short-window-const>)))
((numba_indicators_nan.adx(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], window=<window-const>) < <digit><aug-digit>) & (price_data['btc_high'] > signals.moving_max(price_data['btc_high'], window=<window-const>)))
((numba_indicators_nan.adx(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], window=<window-const>) > <digit><aug-digit>) & (price_data['btc_low'] > signals.moving_min(price_data['btc_low'], window=<window-const>)))
((numba_indicators_nan.adx(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], window=<window-const>) > <digit><aug-digit>) & (numba_indicators_nan.adx(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], window=<window-const>) < <digit><aug-digit>) & (price_data['btc_close'] < signals.moving_percentile(price_data['btc_close'], window=<window-const>, percentile=<perc-const>)))
((numba_indicators_nan.adx(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], window=<window-const>) > <digit><aug-digit>) & (numba_indicators_nan.adx(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], window=<window-const>) < <digit><aug-digit>) & (price_data['btc_close'] > signals.moving_percentile(price_data['btc_close'], window=<window-const>, percentile=<perc-const>)))
((numba_indicators_nan.relative_strength_index(prices=price_data['btc_close'], window=<window-const>) < <digit><aug-digit>) & (price_data['btc_close'] > numba_indicators_nan.moving_average(prices=price_data['btc_close'], window=<window-const>)))
((numba_indicators_nan.relative_strength_index(prices=price_data['btc_close'], window=<window-const>) > <digit><aug-digit>) & (price_data['btc_close'] < numba_indicators_nan.moving_average(prices=price_data['btc_close'], window=<window-const>)))
((numba_indicators_nan.moving_average(prices=price_data['btc_close'], window=<short-window-const>) > numba_indicators_nan.moving_average(prices=price_data['btc_close'], window=<long-window-const>)) & (price_data['btc_close'] < (price_data['btc_low'] + <int-const>)))
((numba_indicators_nan.moving_average(prices=price_data['btc_close'], window=<short-window-const>) < numba_indicators_nan.moving_average(prices=price_data['btc_close'], window=<long-window-const>)) & (price_data['btc_close'] > (price_data['btc_high'] - <int-const>)))
((price_data['day_of_week'] == <week-day>) & (price_data['btc_high'] == signals.moving_max(price_data['btc_high'], window=<window-const>)) & (price_data['btc_close'] == signals.moving_max(price_data['btc_close'], window=<window-const>)) & (numba_indicators_nan.average_true_range(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], window=<window-const>) < <int-const>))
((price_data['day_of_week'] == <week-day>) & (price_data['btc_low'] == signals.moving_min(price_data['btc_low'], window=<window-const>)) & (price_data['btc_close'] == signals.moving_min(price_data['btc_close'], window=<window-const>)) & (numba_indicators_nan.average_true_range(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], window=<window-const>) < <int-const>))
(price_data['btc_close'] == signals.moving_min(price_data['btc_close'], window=<window-const>))
(price_data['btc_close'] == signals.moving_max(price_data['btc_close'], window=<window-const>))
((get_lag(price_data['btc_low'], lag=<lag-steps>) > price_data['btc_high']) & (price_data['btc_high'] > get_lag(price_data['btc_low'], lag=<lag-steps>)) & (price_data['btc_low'] > get_lag(price_data['btc_low'], lag=<lag-steps>)) & (get_lag(price_data['btc_low'], lag=<lag-steps>) > get_lag(price_data['btc_low'], lag=<lag-steps>)))
((get_lag(price_data['btc_low'], lag=3) > price_data['btc_high']) & (price_data['btc_high'] > get_lag(price_data['btc_low'], lag=1)) & (price_data['btc_low'] > get_lag(price_data['btc_low'], lag=2)) & (get_lag(price_data['btc_low'], lag=1) > get_lag(price_data['btc_low'], lag=2)))
((get_lag(price_data['btc_high'], lag=<lag-steps>) < price_data['btc_low']) & (price_data['btc_low'] < get_lag(price_data['btc_high'], lag=<lag-steps>)) & (price_data['btc_high'] < get_lag(price_data['btc_high'], lag=<lag-steps>)) & (get_lag(price_data['btc_high'], lag=<lag-steps>) < get_lag(price_data['btc_high'], lag=<lag-steps>)))
((get_lag(price_data['btc_high'], lag=3) < price_data['btc_low']) & (price_data['btc_low'] < get_lag(price_data['btc_high'], lag=1)) & (price_data['btc_high'] < get_lag(price_data['btc_high'], lag=2)) & (get_lag(price_data['btc_high'], lag=1) < get_lag(price_data['btc_high'], lag=2)))
(((price_data['btc_close'] - signals.moving_min(price_data['btc_low'], window=2)) / (signals.moving_max(price_data['btc_high'], window=2) - signals.moving_min(price_data['btc_low'], window=2))) > 0.<digit>)
(((price_data['btc_close'] - signals.moving_min(price_data['btc_low'], window=2)) / (signals.moving_max(price_data['btc_high'], window=2) - signals.moving_min(price_data['btc_low'], window=2))) < 0.<digit>)
((numba_indicators_nan.exponential_moving_average(prices=price_data['btc_close'], window=<short-window-const>) > numba_indicators_nan.exponential_moving_average(prices=price_data['btc_close'], window=<long-window-const>)) & (get_lag(numba_indicators_nan.exponential_moving_average(prices=price_data['btc_close'], window=<short-window-const>), lag=1) < get_lag(numba_indicators_nan.exponential_moving_average(prices=price_data['btc_close'], window=<long-window-const>), lag=1)))
((numba_indicators_nan.exponential_moving_average(prices=price_data['btc_close'], window=<short-window-const>) < numba_indicators_nan.exponential_moving_average(prices=price_data['btc_close'], window=<long-window-const>)) & (get_lag(numba_indicators_nan.exponential_moving_average(prices=price_data['btc_close'], window=<short-window-const>), lag=1) > get_lag(numba_indicators_nan.exponential_moving_average(prices=price_data['btc_close'], window=<long-window-const>), lag=1)))
((numba_indicators_nan.exponential_moving_average(prices=price_data['btc_close'], window=10) > numba_indicators_nan.exponential_moving_average(prices=price_data['btc_close'], window=40)) & (get_lag(numba_indicators_nan.exponential_moving_average(prices=price_data['btc_close'], window=10), lag=1) < get_lag(numba_indicators_nan.exponential_moving_average(prices=price_data['btc_close'], window=40), lag=1)))
((numba_indicators_nan.exponential_moving_average(prices=price_data['btc_close'], window=10) < numba_indicators_nan.exponential_moving_average(prices=price_data['btc_close'], window=40)) & (get_lag(numba_indicators_nan.exponential_moving_average(prices=price_data['btc_close'], window=10), lag=1) > get_lag(numba_indicators_nan.exponential_moving_average(prices=price_data['btc_close'], window=40), lag=1)))
(numba_indicators_nan.stochastic_oscillator_kd(close_prices=price_data['btc_close'], high_prices=price_data['btc_high'], low_prices=price_data['btc_low'], period=8, smooth_period=22)[0] > numba_indicators_nan.stochastic_oscillator_kd(close_prices=price_data['btc_close'], high_prices=price_data['btc_high'], low_prices=price_data['btc_low'], period=8, smooth_period=22)[1])
(numba_indicators_nan.stochastic_oscillator_kd(close_prices=price_data['btc_close'], high_prices=price_data['btc_high'], low_prices=price_data['btc_low'], period=8, smooth_period=22)[0] < numba_indicators_nan.stochastic_oscillator_kd(close_prices=price_data['btc_close'], high_prices=price_data['btc_high'], low_prices=price_data['btc_low'], period=8, smooth_period=22)[1])
(numba_indicators_nan.money_flow_index(high_prices=price_data['btc_high'], low_prices=price_data['btc_low'], close_prices=price_data['btc_close'], volumes=price_data['btc_volume'], period=14) > 20)
(numba_indicators_nan.money_flow_index(high_prices=price_data['btc_high'], low_prices=price_data['btc_low'], close_prices=price_data['btc_close'], volumes=price_data['btc_volume'], period=14) < 80)
((price_data['btc_close'] > numba_indicators_nan.bollinger_bands(prices=price_data['btc_close'], window=20, num_std_dev=2)[2]) & (price_data['btc_close'] > get_lag(price_data['btc_close'], lag=10)))
((price_data['btc_close'] < numba_indicators_nan.bollinger_bands(prices=price_data['btc_close'], window=20, num_std_dev=2)[1]) & (price_data['btc_close'] < get_lag(price_data['btc_close'], lag=10)))
((price_data['btc_close'] > (numba_indicators_nan.moving_average(prices=price_data['btc_close'], window=20) - 2 * numba_indicators_nan.average_true_range(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], window=20))) & (price_data['btc_close'] > get_lag(price_data['btc_close'], lag=10)))
((price_data['btc_close'] < (numba_indicators_nan.moving_average(prices=price_data['btc_close'], window=20) + 2 * numba_indicators_nan.average_true_range(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], window=20))) & (price_data['btc_close'] < get_lag(price_data['btc_close'], lag=10)))
((numba_indicators_nan.adx(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], window=14) > 25) & (numba_indicators_nan.relative_strength_index(prices=price_data['btc_close'], window=14) < 50) & (price_data['btc_close'] < get_lag(price_data['btc_close'], lag=20)) & (price_data['btc_close'] > get_lag(price_data['btc_close'], lag=10)))
((numba_indicators_nan.adx(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], window=14) > 25) & (numba_indicators_nan.relative_strength_index(prices=price_data['btc_close'], window=14) > 50) & (price_data['btc_close'] > get_lag(price_data['btc_close'], lag=20)) & (price_data['btc_close'] < get_lag(price_data['btc_close'], lag=10)))
((numba_indicators_nan.adx(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], window=14) > 20) & (price_data['btc_close'] > get_lag(price_data['btc_close'], lag=20)))
((numba_indicators_nan.adx(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], window=14) > 20) & (price_data['btc_close'] < get_lag(price_data['btc_close'], lag=20)))
((get_lag(price_data['btc_open'], lag=1) > price_data['btc_high']) & (price_data['btc_open'] > get_lag(price_data['btc_close'], lag=1)) & (get_lag(price_data['btc_close'], lag=1) > get_lag(price_data['btc_low'], lag=1)) & (get_lag(price_data['btc_low'], lag=1) > price_data['btc_close']))
((get_lag(price_data['btc_open'], lag=1) < price_data['btc_low']) & (price_data['btc_open'] < get_lag(price_data['btc_close'], lag=1)) & (get_lag(price_data['btc_close'], lag=1) < get_lag(price_data['btc_high'], lag=1)) & (get_lag(price_data['btc_high'], lag=1) < price_data['btc_close']))
((get_lag(price_data['btc_low'], lag=3) > price_data['btc_high']) & (price_data['btc_high'] > get_lag(price_data['btc_low'], lag=1)) & (get_lag(price_data['btc_low'], lag=1) > get_lag(price_data['btc_low'], lag=2)) & (price_data['btc_close'] > get_lag(price_data['btc_close'], lag=1)))
((get_lag(price_data['btc_high'], lag=3) < price_data['btc_low']) & (price_data['btc_low'] < get_lag(price_data['btc_high'], lag=1)) & (get_lag(price_data['btc_high'], lag=1) < get_lag(price_data['btc_high'], lag=2)) & (price_data['btc_close'] < get_lag(price_data['btc_close'], lag=1)))
((get_lag(price_data['btc_close'], lag=1) > get_lag(price_data['btc_close'], lag=3)) & (price_data['btc_close'] > get_lag(price_data['btc_close'], lag=2)) & (get_lag(price_data['btc_close'], lag=2) > get_lag(price_data['btc_close'], lag=1))) 
((get_lag(price_data['btc_close'], lag=1) < get_lag(price_data['btc_close'], lag=3)) & (price_data['btc_close'] < get_lag(price_data['btc_close'], lag=2)) & (get_lag(price_data['btc_close'], lag=2) < get_lag(price_data['btc_close'], lag=1)))
((get_lag(price_data['btc_high'], lag=2) > get_lag(price_data['btc_high'], lag=1)) & (get_lag(price_data['btc_low'], lag=2) < get_lag(price_data['btc_low'], lag=1)) & (price_data['btc_close'] > get_lag(price_data['btc_high'], lag=2)))
((get_lag(price_data['btc_low'], lag=2) < get_lag(price_data['btc_low'], lag=1)) & (get_lag(price_data['btc_high'], lag=2) > get_lag(price_data['btc_high'], lag=1)) & (price_data['btc_close'] < get_lag(price_data['btc_low'], lag=2)))
((get_lag(price_data['btc_close'], lag=1) < get_lag(price_data['btc_close'], lag=2)) & (get_lag(price_data['btc_close'], lag=2) < get_lag(price_data['btc_close'], lag=5)) & (get_lag(price_data['btc_close'], lag=5) < get_lag(price_data['btc_close'], lag=3)) & (get_lag(price_data['btc_close'], lag=3) < get_lag(price_data['btc_close'], lag=4)))
((get_lag(price_data['btc_close'], lag=1) > get_lag(price_data['btc_close'], lag=2)) & (get_lag(price_data['btc_close'], lag=2) > get_lag(price_data['btc_close'], lag=5)) & (get_lag(price_data['btc_close'], lag=5) > get_lag(price_data['btc_close'], lag=3)) & (get_lag(price_data['btc_close'], lag=3) > get_lag(price_data['btc_close'], lag=4)))
(numba_indicators_nan.moving_average(prices=commodity_channel_index(high_prices=price_data['btc_high'], low_prices=price_data['btc_low'], close_prices=price_data['btc_close'], period=14), window=9) >= 100)
(numba_indicators_nan.moving_average(prices=commodity_channel_index(high_prices=price_data['btc_high'], low_prices=price_data['btc_low'], close_prices=price_data['btc_close'], period=14), window=9) <= -100)
((numba_indicators_nan.bull_bar_tail_rolling(close_prices=price_data['btc_close'], open_prices=price_data['btc_open'], high_prices=price_data['btc_high'], low_prices=price_data['btc_low'], period=10) > numba_indicators_nan.bear_bar_tail_rolling(close_prices=price_data['btc_close'], open_prices=price_data['btc_open'], high_prices=price_data['btc_high'], low_prices=price_data['btc_low'], period=10)) & (numba_indicators_nan.bull_bar_tail_rolling(close_prices=price_data['btc_close'], open_prices=price_data['btc_open'], high_prices=price_data['btc_high'], low_prices=price_data['btc_low'], period=10) > 5))
((numba_indicators_nan.bull_bar_tail_rolling(close_prices=price_data['btc_close'], open_prices=price_data['btc_open'], high_prices=price_data['btc_high'], low_prices=price_data['btc_low'], period=10) < numba_indicators_nan.bear_bar_tail_rolling(close_prices=price_data['btc_close'], open_prices=price_data['btc_open'], high_prices=price_data['btc_high'], low_prices=price_data['btc_low'], period=10)) & (numba_indicators_nan.bear_bar_tail_rolling(close_prices=price_data['btc_close'], open_prices=price_data['btc_open'], high_prices=price_data['btc_high'], low_prices=price_data['btc_low'], period=10) > 5))
((price_data['btc_close'] > signals.moving_max(get_lag(price_data['btc_high'], lag=1), window=<window-const>)) & (price_data['btc_close'] > get_lag(price_data['btc_close'], lag=1)) & (price_data['btc_close'] > get_lag(price_data['btc_close'], lag=3)) & (get_lag(price_data['btc_close'], lag=1) > get_lag(price_data['btc_close'], lag=2)))
((price_data['btc_close'] < signals.moving_min(get_lag(price_data['btc_low'], lag=1), window=<window-const>)) & (price_data['btc_close'] < get_lag(price_data['btc_close'], lag=1)) & (price_data['btc_close'] < get_lag(price_data['btc_close'], lag=3)) & (get_lag(price_data['btc_close'], lag=1) < get_lag(price_data['btc_close'], lag=2)))
((numba_indicators_nan.awesome_oscillator(high_prices=price_data['btc_high'], low_prices=price_data['btc_low'], short_period=5, long_period=7) < get_lag(numba_indicators_nan.awesome_oscillator(high_prices=price_data['btc_high'], low_prices=price_data['btc_low'], short_period=5, long_period=7), lag=1)) & (price_data['btc_high'] > get_lag(price_data['btc_high'], lag=1)) & (((price_data['btc_close'] - price_data['btc_low'])/(price_data['btc_high'] - price_data['btc_low'])) < 0.5))
((numba_indicators_nan.awesome_oscillator(high_prices=price_data['btc_high'], low_prices=price_data['btc_low'], short_period=5, long_period=7) > get_lag(numba_indicators_nan.awesome_oscillator(high_prices=price_data['btc_high'], low_prices=price_data['btc_low'], short_period=5, long_period=7), lag=1)) & (price_data['btc_low'] < get_lag(price_data['btc_low'], lag=1)) & (((price_data['btc_close'] - price_data['btc_low'])/(price_data['btc_high'] - price_data['btc_low'])) > 0.5))
((price_data['hour'] >= 21) & (price_data['hour'] <= 23) & (price_data['btc_close'] > get_lag(price_data['btc_close'], lag=<lag-steps>)))
((price_data['hour'] >= 22) & (price_data['hour'] <= 23) & (price_data['btc_close'] < get_lag(price_data['btc_close'], lag=<lag-steps>)))
(((get_lag(price_data['btc_high'], lag=1) - get_lag(price_data['btc_low'], lag=1)) < (get_lag(price_data['btc_high'], lag=2) - get_lag(price_data['btc_low'], lag=2))) & (price_data['btc_close'] == signals.moving_min(price_data['btc_close'], window=<window-const>)))
(((get_lag(price_data['btc_high'], lag=1) - get_lag(price_data['btc_low'], lag=1)) < (get_lag(price_data['btc_high'], lag=2) - get_lag(price_data['btc_low'], lag=2))) & (price_data['btc_close'] == signals.moving_max(price_data['btc_close'], window=<window-const>)))
(numba_indicators_nan.ultimate_oscillator(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], period1=7, period2=14, period3=28) == signals.moving_max(numba_indicators_nan.ultimate_oscillator(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], period1=7, period2=14, period3=28), window=<window-const>))
(numba_indicators_nan.ultimate_oscillator(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], period1=7, period2=14, period3=28) == signals.moving_min(numba_indicators_nan.ultimate_oscillator(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], period1=7, period2=14, period3=28), window=<window-const>))
(price_data['day_of_week'] == <week-day>)
((price_data['btc_close'] > get_lag(price_data['btc_close'], lag=<short-window-const>)) & (price_data['btc_close'] < get_lag(price_data['btc_close'], lag=<long-window-const>)))
((price_data['btc_close'] < get_lag(price_data['btc_close'], lag=<short-window-const>)) & (price_data['btc_close'] > get_lag(price_data['btc_close'], lag=<long-window-const>)))
((price_data['btc_close'] > numba_indicators_nan.moving_average(prices=price_data['btc_close'], window=10)) & ((price_data['btc_close'] < signals.moving_percentile(price_data['btc_close'], window=15, percentile=0.1)) | ((price_data['btc_close'] < get_lag(price_data['btc_close'], lag=1)) & (get_lag(price_data['btc_close'], lag=1) < get_lag(price_data['btc_close'], lag=2)) & (get_lag(price_data['btc_close'], lag=2) < get_lag(price_data['btc_close'], lag=3)))))
((price_data['btc_close'] < numba_indicators_nan.moving_average(prices=price_data['btc_close'], window=10)) & ((price_data['btc_close'] > signals.moving_percentile(price_data['btc_close'], window=15, percentile=0.1)) | ((price_data['btc_close'] > get_lag(price_data['btc_close'], lag=1)) & (get_lag(price_data['btc_close'], lag=1) > get_lag(price_data['btc_close'], lag=2)) & (get_lag(price_data['btc_close'], lag=2) > get_lag(price_data['btc_close'], lag=3)))))
(((price_data['btc_high'] - price_data['btc_low']) > (2 * numba_indicators_nan.rolling_volatility((price_data['btc_high'] - price_data['btc_low']), window_size=5) + numba_indicators_nan.moving_average(prices=(price_data['btc_high'] - price_data['btc_low']), window=5))) & (price_data['btc_close'] > get_lag(price_data['btc_close'], lag=10)))
(((price_data['btc_high'] - price_data['btc_low']) > (2 * numba_indicators_nan.rolling_volatility((price_data['btc_high'] - price_data['btc_low']), window_size=5) + numba_indicators_nan.moving_average(prices=(price_data['btc_high'] - price_data['btc_low']), window=5))) & (price_data['btc_close'] < get_lag(price_data['btc_close'], lag=10)))
((price_data['btc_volume'] < numba_indicators_nan.moving_average(prices=price_data['btc_close'], window=5)) & (price_data['btc_close'] == signals.moving_min(price_data['btc_close'], window=5)))
((price_data['btc_volume'] < numba_indicators_nan.moving_average(prices=price_data['btc_close'], window=5)) & (price_data['btc_close'] == signals.moving_max(price_data['btc_close'], window=5)))
((get_lag(price_data['btc_low'], lag=3) > get_lag(price_data['btc_low'], lag=2)) & (get_lag(price_data['btc_low'], lag=2) > get_lag(price_data['btc_low'], lag=1)) & (price_data['btc_close'] > get_lag(price_data['btc_high'], lag=1)))
((get_lag(price_data['btc_high'], lag=3) < get_lag(price_data['btc_high'], lag=2)) & (get_lag(price_data['btc_high'], lag=2) < get_lag(price_data['btc_high'], lag=1)) & (price_data['btc_close'] < get_lag(price_data['btc_low'], lag=1)))'''

exit_txt = '''(price_data['btc_close'] == signals.moving_min(price_data['btc_close'], window=<window-const>))
(price_data['btc_close'] == signals.moving_max(price_data['btc_close'], window=<window-const>))
(numba_indicators_nan.average_true_range(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], window=<window-const>) > <short-window-const>)
(numba_indicators_nan.average_true_range(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], window=<window-const>) < <short-window-const>)
(price_data['btc_close'] > signals.moving_percentile(price_data['btc_close'], window=<window-const>, percentile=<perc-const>))
(price_data['btc_close'] < signals.moving_percentile(price_data['btc_close'], window=<window-const>, percentile=<perc-const>))
((price_data['btc_close'] > get_lag(price_data['btc_close'], lag=1)) & (get_lag(price_data['btc_close'], lag=1) > get_lag(price_data['btc_close'], lag=2)) & (get_lag(price_data['btc_close'], lag=2) > get_lag(price_data['btc_close'], lag=3)))
((price_data['btc_close'] < get_lag(price_data['btc_close'], lag=1)) & (get_lag(price_data['btc_close'], lag=1) < get_lag(price_data['btc_close'], lag=2)) & (get_lag(price_data['btc_close'], lag=2) < get_lag(price_data['btc_close'], lag=3)))
(price_data['btc_high'] < signals.moving_max(price_data['btc_high'], window=<window-const>))
(price_data['btc_high'] > signals.moving_max(price_data['btc_high'], window=<window-const>))
(price_data['btc_low'] < signals.moving_min(price_data['btc_low'], window=<window-const>))
(price_data['btc_low'] > signals.moving_min(price_data['btc_low'], window=<window-const>))
(price_data['btc_low'] < signals.moving_max(price_data['btc_low'], window=<window-const>))
(price_data['btc_low'] > signals.moving_max(price_data['btc_low'], window=<window-const>))
(price_data['btc_high'] > signals.moving_min(price_data['btc_high'], window=<window-const>))
(price_data['btc_high'] < signals.moving_min(price_data['btc_high'], window=<window-const>))
(numba_indicators_nan.true_range(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], window=<window-const>) > <aug-digit> * numba_indicators_nan.average_true_range(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], window=<window-const>))
(numba_indicators_nan.true_range(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], window=<window-const>) < <aug-digit> * numba_indicators_nan.average_true_range(high=price_data['btc_high'], low=price_data['btc_low'], close=price_data['btc_close'], window=<window-const>))'''

entry_blueprint_list = entry_txt.split('\n')
exit_blueprint_list = exit_txt.split('\n')

entry_params_list = []

for i in entry_blueprint_list:
    entry_params_list.extend(re.findall(r'\<([a-z-]+)\>', i))

entry_params_list = list(set(entry_params_list))

exit_params_list = []

for i in exit_blueprint_list:
    exit_params_list.extend(re.findall(r'\<([a-z-]+)\>', i))

exit_params_list = list(set(exit_params_list))

params_list = entry_params_list.copy()
params_list.extend(exit_params_list)
params_list = list(set(params_list))

params_dict = {
    'int-const': [10, 21, 36, 48, 59, 302], #list(np.arange(1, 500)),
    'long-window-const': [150, 201, 261, 311, 457, 500], #list(np.arange(100, 500)),
    'digit': list(np.arange(1, 10)),
    'lag-steps': [1, 5, 7, 10, 50], #list(np.arange(1, 500)),
    'perc-const': list(np.linspace(0.01, 0.99, 10)),
    'short-window-const': [5, 7, 10, 15], #list(np.arange(1, 100)),
    'window-const': [10, 15, 25, 35, 45, 55, 100],#list(np.arange(1, 500)),
    'aug-digit': list(np.arange(0, 10)),
    'week-day': list(np.arange(1, 8))
}

str_comb_list = list(itertools.product(entry_blueprint_list, entry_blueprint_list, exit_blueprint_list, exit_blueprint_list))

def place_values(txt_str, str_params, param_values, idx=0):

    for i_param in range(len(str_params)):
        temp_split_str = txt_str.split(str_params[i_param])
        temp_prev_txt = ''.join([temp_split_str[0], str_params[i_param], '>'])
        temp_split_str[0] = temp_split_str[0][:-1]
        temp_split_str[1] = temp_split_str[1][1:]
        temp_split_str.insert(1, str(param_values[idx]))
        temp_new_txt = ''.join(temp_split_str[:2])
        txt_str = txt_str.replace(temp_prev_txt, temp_new_txt)
        idx += 1
    
    return txt_str, idx

buy_list = []
sell_list = []
exit_buy_list = []
exit_sell_list = []

for temp_str_comb in tqdm(str_comb_list[:1000]):

    buy = temp_str_comb[0]
    sell = temp_str_comb[1]
    exit_buy = temp_str_comb[2]
    exit_sell = temp_str_comb[3]

    buy_params = re.findall(r'\<([a-z-]+)\>', buy)
    buy_params = [p for p in buy_params if p in params_dict.keys()]

    sell_params = re.findall(r'\<([a-z-]+)\>', sell)
    sell_params = [p for p in sell_params if p in params_dict.keys()]

    exit_buy_params = re.findall(r'\<([a-z-]+)\>', exit_buy)
    exit_buy_params = [p for p in exit_buy_params if p in params_dict.keys()]

    exit_sell_params = re.findall(r'\<([a-z-]+)\>', exit_sell)
    exit_sell_params = [p for p in exit_sell_params if p in params_dict.keys()]

    temp_params = buy_params.copy()
    temp_params.extend(sell_params)
    temp_params.extend(exit_buy_params)
    temp_params.extend(exit_sell_params)

    exec_dict = {'params_dict': params_dict}

    temp_txt_itertools = "import itertools\ntemp_params_comb_list = list(itertools.product("
    temp_txt_itertools += ", ".join([f"params_dict['{p}']" for p in temp_params])
    temp_txt_itertools += "))"

    exec(temp_txt_itertools, exec_dict)

    temp_params_comb_list = exec_dict['temp_params_comb_list']

    for temp_params_comb in temp_params_comb_list:

        temp_buy, idx = place_values(txt_str=buy, str_params=buy_params, param_values=temp_params_comb, idx=0)
        temp_sell, idx = place_values(txt_str=sell, str_params=sell_params, param_values=temp_params_comb, idx=idx)
        temp_exit_buy, idx = place_values(txt_str=exit_buy, str_params=exit_buy_params, param_values=temp_params_comb, idx=idx)
        temp_exit_sell, idx = place_values(txt_str=exit_sell, str_params=exit_sell_params, param_values=temp_params_comb, idx=idx)

        buy_list.append(temp_buy)
        sell_list.append(temp_sell)
        exit_buy_list.append(temp_exit_buy)
        exit_sell_list.append(temp_exit_sell)

    gc.collect()

df_str = pd.DataFrame()
df_str['buy'] = buy_list
df_str['exit_buy'] = exit_buy_list
df_str['sell'] = sell_list
df_str['exit_sell'] = exit_sell_list
df_str['strategy'] = [f'strategy{i}' for i in range(len(buy_list))]
df_str.to_csv('kjd_str.csv', index=False)