import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import gc
import os
from tqdm import tqdm
from .testing_func import (calculate_mean_win_perc_entry_testing, calculate_mean_win_perc_exit_testing,
                          calculate_mean_win_perc_core_testing, calculate_mean_performance, 
                          calculate_mc_performance, get_entry_win_pc_df, get_exit_win_pc_df,
                          get_core_win_pc_df, get_perf_df, get_mc_df)
from .filter_strategies2 import generate_fold_data, change_frequency
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')

def generate_fold_data_valid(data_path, fold_size, time_freq, fold):

    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.sort_values('datetime', ascending=True, inplace=True)
    df.reset_index(inplace=True, drop=True)

    years_list = list(df['datetime'].dt.year.unique())

    df_freq = change_frequency(data=df, freq=f'{time_freq}min', instrument_name='btc')

    valid_fold_year = years_list[fold - 1] + fold_size

    temp_df_valid = df_freq[df_freq['datetime'].dt.year==valid_fold_year]
    temp_df_valid.reset_index(drop=True, inplace=True)

    return temp_df_valid

def generate_fold_data_test(data_path, fold_size, time_freq, fold):

    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.sort_values('datetime', ascending=True, inplace=True)
    df.reset_index(inplace=True, drop=True)

    years_list = list(df['datetime'].dt.year.unique())

    df_freq = change_frequency(data=df, freq=f'{time_freq}min', instrument_name='btc')

    test_fold_year = years_list[fold - 1] + fold_size + 1

    temp_df_test = df_freq[df_freq['datetime'].dt.year==test_fold_year]
    temp_df_test.reset_index(drop=True, inplace=True)

    return temp_df_test

def downside_deviation(returns, target):
    downside_returns = returns[returns < target]
    deviations = downside_returns - target
    return deviations

def inverse_variance_weights(cov_matrix):
    inv_var = 1 / np.diag(cov_matrix)
    weights = inv_var / np.sum(inv_var)
    return weights

def calculate_portfolio_frontier(df, annualized_log_return, var_matrix, downside_cov_matrix, annual_factor):

    port_returns = []
    port_volatility = []
    port_weights = []
    port_downside_volatility = []

    n_strs = df.shape[1]

    n_ports = 10000

    for _ in range(n_ports):

        temp_weights = np.random.random(n_strs)
        temp_weights = temp_weights / np.sum(temp_weights)
        port_weights.append(temp_weights)

        temp_port_return = np.dot(temp_weights, annualized_log_return.values)
        port_returns.append(temp_port_return)

        var = var_matrix.mul(temp_weights, axis=0).mul(temp_weights, axis=1).sum().sum()
        sd = np.sqrt(var)
        ann_sd = sd * np.sqrt(annual_factor)
        port_volatility.append(ann_sd)

        downside_var = downside_cov_matrix.mul(temp_weights, axis=0).mul(temp_weights, axis=1).sum().sum()
        downside_sd = np.sqrt(downside_var)
        ann_downside_sd = downside_sd * np.sqrt(annual_factor)
        port_downside_volatility.append(ann_downside_sd)

    return port_returns, port_volatility, port_weights, port_downside_volatility

def create_txt_code1(buy_signal_txt, buy_exit_txt, sell_signal_txt, sell_exit_txt, 
                     fee=0.015, slippage=0.00005, inv_amount=700000, trade_size=0.5, max_lag=99):

    text_code = f'''import os
CUR_DIR = os.getcwd()
# os.chdir('src')
#import pandas as pd
import numpy as np
import gc
from fitness.indicators import numba_indicators_nan, signals
from fitness.performance.helper_func import merge_buy_sell_pnl, get_drawdowns, get_pnl, get_lag
from fitness.performance.helper_func import trading_signals_buy, trading_signals_sell, change_exit
# os.chdir(CUR_DIR)
#from numba import njit
COMMISSION = {fee}
SLIPPAGE = {slippage}
AVAILABLE_CAPITAL = {inv_amount}
TRADE_SIZE = {trade_size}
MAX_LAG = {max_lag}
try:
    buy_idxs, buy_exit_idxs = trading_signals_buy(buy_signal={buy_signal_txt}, exit_signal={buy_exit_txt})
except:
    buy_idxs, buy_exit_idxs = [], []
try:
    sell_idxs, sell_exit_idxs = trading_signals_sell(sell_signal={sell_signal_txt}, exit_signal={sell_exit_txt})
except:
    sell_idxs, sell_exit_idxs = [], []
# if (len(buy_idxs) == 0 or len(buy_exit_idxs) == 0) and (len(sell_idxs) == 0 or len(sell_exit_idxs) == 0):
#     fitness = -9999999
#     avg_drawdown = -9999999
# else:
try:
    buy_idxs, buy_exit_idxs, sell_idxs, sell_exit_idxs = change_exit(buy_idxs, buy_exit_idxs, sell_idxs, sell_exit_idxs)
except:
    pass
if (len(buy_idxs) == 0 or len(buy_exit_idxs) == 0) and (len(sell_idxs) == 0 or len(sell_exit_idxs) == 0):
    fitness = np.nan
    avg_drawdown = np.nan
else:
    buy_idxs = np.array(buy_idxs)
    sell_idxs = np.array(sell_idxs)
    open_prices = price_data['btc_open']
    # pnl_mren_arr, max_dd_mren_arr = get_monkey_test_results(open_prices, buy_idxs, sell_idxs, COMMISSION, SLIPPAGE, AVAILABLE_CAPITAL, TRADE_SIZE)
    buy_prices = open_prices[np.isin(np.arange(len(open_prices)), buy_idxs)]
    buy_exit_prices = open_prices[np.isin(np.arange(len(open_prices)), buy_exit_idxs)]
    sell_prices = open_prices[np.isin(np.arange(len(open_prices)), sell_idxs)]
    sell_exit_prices = open_prices[np.isin(np.arange(len(open_prices)), sell_exit_idxs)]
    buy_arr = get_pnl(buy_exit_prices, buy_prices, COMMISSION, SLIPPAGE, AVAILABLE_CAPITAL, TRADE_SIZE, 1)
    buy_pnl = np.sum(buy_arr)
    sell_arr = get_pnl(sell_exit_prices, sell_prices, COMMISSION, SLIPPAGE, AVAILABLE_CAPITAL, TRADE_SIZE, 0)
    sell_pnl = np.sum(sell_arr)
    all_arr = merge_buy_sell_pnl(buy_idxs, sell_idxs, buy_arr, sell_arr)
    total_pnl = buy_pnl + sell_pnl
    equity_curve_arr = np.cumsum(all_arr)
    drawdowns = get_drawdowns(equity_curve_arr)
    if len(drawdowns[drawdowns!=0]) != 0:
        avg_drawdown = np.sum(drawdowns[drawdowns!=0]) / len(drawdowns[drawdowns!=0])
        fitness = total_pnl / avg_drawdown
    else:
        fitness = np.nan
        avg_drawdown = np.nan
gc.collect()'''
    
    return text_code

def create_txt_code1_vbt(buy_signal_txt, buy_exit_txt, sell_signal_txt, sell_exit_txt, 
                     fee=0.015, slippage=0.00005, inv_amount=700000, trade_size=0.5, max_lag=99):

    text_code = f'''import os
CUR_DIR = os.getcwd()
# os.chdir('src')
import pandas as pd
import numpy as np
import vectorbt as vbt
import gc
from fitness.indicators import numba_indicators_nan, signals
from fitness.performance.helper_func import merge_buy_sell_pnl, get_drawdowns, get_pnl, get_lag
from fitness.performance.helper_func import trading_signals_buy, trading_signals_sell, change_exit
# os.chdir(CUR_DIR)
#from numba import njit
COMMISSION = {fee}
SLIPPAGE = {slippage}
AVAILABLE_CAPITAL = {inv_amount}
TRADE_SIZE = {trade_size}
MAX_LAG = {max_lag}
try:
    buy_idxs, buy_exit_idxs = trading_signals_buy(buy_signal={buy_signal_txt}, exit_signal={buy_exit_txt})
except:
    buy_idxs, buy_exit_idxs = [], []
try:
    sell_idxs, sell_exit_idxs = trading_signals_sell(sell_signal={sell_signal_txt}, exit_signal={sell_exit_txt})
except:
    sell_idxs, sell_exit_idxs = [], []
# if (len(buy_idxs) == 0 or len(buy_exit_idxs) == 0) and (len(sell_idxs) == 0 or len(sell_exit_idxs) == 0):
#     fitness = -9999999
#     avg_drawdown = -9999999
# else:
try:
    buy_idxs, buy_exit_idxs, sell_idxs, sell_exit_idxs = change_exit(buy_idxs, buy_exit_idxs, sell_idxs, sell_exit_idxs)
except:
    pass
if (len(buy_idxs) == 0 or len(buy_exit_idxs) == 0) and (len(sell_idxs) == 0 or len(sell_exit_idxs) == 0):
    fitness = np.nan
else:
    buy_entries = np.array([1 if i in buy_idxs else 0 for i in range(len(price_data['btc_open']))])
    buy_exits = np.array([1 if i in buy_exit_idxs else 0 for i in range(len(price_data['btc_open']))])
    sell_entries = np.array([1 if i in sell_idxs else 0 for i in range(len(price_data['btc_open']))])
    sell_exits = np.array([1 if i in sell_exit_idxs else 0 for i in range(len(price_data['btc_open']))])
    price_data_open = pd.Series(price_data['btc_open'].reshape(-1, ), index=pd.to_datetime(price_data['datetime']))
    pf = vbt.Portfolio.from_signals(
        price_data_open, entries=buy_entries, exits=buy_exits, 
        init_cash=AVAILABLE_CAPITAL, fees=COMMISSION, 
        slippage=SLIPPAGE, size=TRADE_SIZE, 
        short_entries=sell_entries, short_exits=sell_exits
    )
    total_return_p = pf.stats()['Total Return [%]']
    max_drawdown_p = pf.stats()['Max Drawdown [%]']
    fitness = total_return_p / max_drawdown_p
    trades = pf.trades.records
    all_arr = trades['pnl'].values
    equity_curve_arr = np.cumsum(all_arr)
    drawdowns = get_drawdowns(equity_curve_arr)
    if len(drawdowns[drawdowns!=0]) != 0:
        avg_drawdown = np.sum(drawdowns[drawdowns!=0]) / len(drawdowns[drawdowns!=0])
    else:
        avg_drawdown = np.nan
gc.collect()'''
    
    return text_code

def create_txt_code_port1(buy_signal_txt, buy_exit_txt, sell_signal_txt, sell_exit_txt, weight,
                     fee=0.015, slippage=0.00005, inv_amount=700000, trade_size=0.5, max_lag=99):

    text_code = f'''import os
CUR_DIR = os.getcwd()
# os.chdir('src')
#import pandas as pd
import numpy as np
import gc
from fitness.indicators import numba_indicators_nan, signals
from fitness.performance.helper_func import merge_buy_sell_pnl, get_drawdowns, get_pnl, get_lag
from fitness.performance.helper_func import trading_signals_buy, trading_signals_sell, change_exit
# os.chdir(CUR_DIR)
#from numba import njit
COMMISSION = {fee}
SLIPPAGE = {slippage}
AVAILABLE_CAPITAL = {inv_amount}
TRADE_SIZE = {trade_size} * {weight}
MAX_LAG = {max_lag}
try:
    buy_idxs, buy_exit_idxs = trading_signals_buy(buy_signal={buy_signal_txt}, exit_signal={buy_exit_txt})
except:
    buy_idxs, buy_exit_idxs = [], []
try:
    sell_idxs, sell_exit_idxs = trading_signals_sell(sell_signal={sell_signal_txt}, exit_signal={sell_exit_txt})
except:
    sell_idxs, sell_exit_idxs = [], []
# if (len(buy_idxs) == 0 or len(buy_exit_idxs) == 0) and (len(sell_idxs) == 0 or len(sell_exit_idxs) == 0):
#     fitness = -9999999
#     avg_drawdown = -9999999
# else:
try:
    buy_idxs, buy_exit_idxs, sell_idxs, sell_exit_idxs = change_exit(buy_idxs, buy_exit_idxs, sell_idxs, sell_exit_idxs)
except:
    pass
if (len(buy_idxs) == 0 or len(buy_exit_idxs) == 0) and (len(sell_idxs) == 0 or len(sell_exit_idxs) == 0):
    fitness = np.nan
    avg_drawdown = np.nan
else:
    buy_idxs = np.array(buy_idxs)
    sell_idxs = np.array(sell_idxs)
    open_prices = price_data['btc_open']
    # pnl_mren_arr, max_dd_mren_arr = get_monkey_test_results(open_prices, buy_idxs, sell_idxs, COMMISSION, SLIPPAGE, AVAILABLE_CAPITAL, TRADE_SIZE)
    buy_prices = open_prices[np.isin(np.arange(len(open_prices)), buy_idxs)]
    buy_exit_prices = open_prices[np.isin(np.arange(len(open_prices)), buy_exit_idxs)]
    sell_prices = open_prices[np.isin(np.arange(len(open_prices)), sell_idxs)]
    sell_exit_prices = open_prices[np.isin(np.arange(len(open_prices)), sell_exit_idxs)]
    buy_arr = get_pnl(buy_exit_prices, buy_prices, COMMISSION, SLIPPAGE, AVAILABLE_CAPITAL, TRADE_SIZE, 1)
    buy_pnl = np.sum(buy_arr)
    sell_arr = get_pnl(sell_exit_prices, sell_prices, COMMISSION, SLIPPAGE, AVAILABLE_CAPITAL, TRADE_SIZE, 0)
    sell_pnl = np.sum(sell_arr)
    all_arr = merge_buy_sell_pnl(buy_idxs, sell_idxs, buy_arr, sell_arr)
    total_pnl = buy_pnl + sell_pnl
    equity_curve_arr = np.cumsum(all_arr)
    drawdowns = get_drawdowns(equity_curve_arr)
    if len(drawdowns[drawdowns!=0]) != 0:
        avg_drawdown = np.sum(drawdowns[drawdowns!=0]) / len(drawdowns[drawdowns!=0])
        fitness = total_pnl / avg_drawdown
    else:
        fitness = np.nan
        avg_drawdown = np.nan
gc.collect()'''
    
    return text_code

def create_txt_code_port1_vbt(buy_signal_txt, buy_exit_txt, sell_signal_txt, sell_exit_txt, weight,
                     fee=0.015, slippage=0.00005, inv_amount=700000, trade_size=0.5, max_lag=99):

    text_code = f'''import os
CUR_DIR = os.getcwd()
# os.chdir('src')
import pandas as pd
import numpy as np
import vectorbt as vbt
import gc
from fitness.indicators import numba_indicators_nan, signals
from fitness.performance.helper_func import merge_buy_sell_pnl, get_drawdowns, get_pnl, get_lag
from fitness.performance.helper_func import trading_signals_buy, trading_signals_sell, change_exit
# os.chdir(CUR_DIR)
#from numba import njit
COMMISSION = {fee}
SLIPPAGE = {slippage}
AVAILABLE_CAPITAL = {inv_amount}
TRADE_SIZE = {trade_size} * {weight}
MAX_LAG = {max_lag}
try:
    buy_idxs, buy_exit_idxs = trading_signals_buy(buy_signal={buy_signal_txt}, exit_signal={buy_exit_txt})
except:
    buy_idxs, buy_exit_idxs = [], []
try:
    sell_idxs, sell_exit_idxs = trading_signals_sell(sell_signal={sell_signal_txt}, exit_signal={sell_exit_txt})
except:
    sell_idxs, sell_exit_idxs = [], []
# if (len(buy_idxs) == 0 or len(buy_exit_idxs) == 0) and (len(sell_idxs) == 0 or len(sell_exit_idxs) == 0):
#     fitness = -9999999
#     avg_drawdown = -9999999
# else:
try:
    buy_idxs, buy_exit_idxs, sell_idxs, sell_exit_idxs = change_exit(buy_idxs, buy_exit_idxs, sell_idxs, sell_exit_idxs)
except:
    pass
if (len(buy_idxs) == 0 or len(buy_exit_idxs) == 0) and (len(sell_idxs) == 0 or len(sell_exit_idxs) == 0):
    fitness = np.nan
else:
    buy_entries = np.array([1 if i in buy_idxs else 0 for i in range(len(price_data['btc_open']))])
    buy_exits = np.array([1 if i in buy_exit_idxs else 0 for i in range(len(price_data['btc_open']))])
    sell_entries = np.array([1 if i in sell_idxs else 0 for i in range(len(price_data['btc_open']))])
    sell_exits = np.array([1 if i in sell_exit_idxs else 0 for i in range(len(price_data['btc_open']))])
    price_data_open = pd.Series(price_data['btc_open'].reshape(-1, ), index=pd.to_datetime(price_data['datetime']))
    pf = vbt.Portfolio.from_signals(
        price_data_open, entries=buy_entries, exits=buy_exits, 
        init_cash=AVAILABLE_CAPITAL, fees=COMMISSION, 
        slippage=SLIPPAGE, size=TRADE_SIZE, 
        short_entries=sell_entries, short_exits=sell_exits
    )
    total_return_p = pf.stats()['Total Return [%]']
    max_drawdown_p = pf.stats()['Max Drawdown [%]']
    fitness = total_return_p / max_drawdown_p
    trades = pf.trades.records
    all_arr = trades['pnl'].values
    equity_curve_arr = np.cumsum(all_arr)
    drawdowns = get_drawdowns(equity_curve_arr)
    if len(drawdowns[drawdowns!=0]) != 0:
        avg_drawdown = np.sum(drawdowns[drawdowns!=0]) / len(drawdowns[drawdowns!=0])
    else:
        avg_drawdown = np.nan
gc.collect()'''
    
    return text_code

def create_txt_code_pnl1(buy_signal_txt, buy_exit_txt, sell_signal_txt, sell_exit_txt, 
                     fee=0.015, slippage=0.00005, inv_amount=700000, trade_size=0.5, max_lag=99):

    text_code = f'''import os
CUR_DIR = os.getcwd()
# os.chdir('src')
#import pandas as pd
import numpy as np
import gc
from fitness.indicators import numba_indicators_nan, signals
from fitness.performance.helper_func import merge_buy_sell_pnl, get_drawdowns, get_pnl, get_lag
from fitness.performance.helper_func import trading_signals_buy, trading_signals_sell, change_exit, get_returns
# os.chdir(CUR_DIR)
#from numba import njit
COMMISSION = {fee}
SLIPPAGE = {slippage}
AVAILABLE_CAPITAL = {inv_amount}
TRADE_SIZE = {trade_size}
MAX_LAG = {max_lag}
try:
    buy_idxs, buy_exit_idxs = trading_signals_buy(buy_signal={buy_signal_txt}, exit_signal={buy_exit_txt})
except:
    buy_idxs, buy_exit_idxs = [], []
try:
    sell_idxs, sell_exit_idxs = trading_signals_sell(sell_signal={sell_signal_txt}, exit_signal={sell_exit_txt})
except:
    sell_idxs, sell_exit_idxs = [], []
# if (len(buy_idxs) == 0 or len(buy_exit_idxs) == 0) and (len(sell_idxs) == 0 or len(sell_exit_idxs) == 0):
#     fitness = -9999999
#     avg_drawdown = -9999999
# else:
try:
    buy_idxs, buy_exit_idxs, sell_idxs, sell_exit_idxs = change_exit(buy_idxs, buy_exit_idxs, sell_idxs, sell_exit_idxs)
except:
    pass
if (len(buy_idxs) == 0 or len(buy_exit_idxs) == 0) and (len(sell_idxs) == 0 or len(sell_exit_idxs) == 0):
    fitness = np.nan
    avg_drawdown = np.nan
else:
    buy_idxs = np.array(buy_idxs)
    sell_idxs = np.array(sell_idxs)
    open_prices = price_data['btc_open']
    # pnl_mren_arr, max_dd_mren_arr = get_monkey_test_results(open_prices, buy_idxs, sell_idxs, COMMISSION, SLIPPAGE, AVAILABLE_CAPITAL, TRADE_SIZE)
    buy_prices = open_prices[np.isin(np.arange(len(open_prices)), buy_idxs)]
    buy_exit_prices = open_prices[np.isin(np.arange(len(open_prices)), buy_exit_idxs)]
    sell_prices = open_prices[np.isin(np.arange(len(open_prices)), sell_idxs)]
    sell_exit_prices = open_prices[np.isin(np.arange(len(open_prices)), sell_exit_idxs)]
    buy_arr = get_pnl(buy_exit_prices, buy_prices, COMMISSION, SLIPPAGE, AVAILABLE_CAPITAL, TRADE_SIZE, 1)
    buy_pnl = np.sum(buy_arr)
    sell_arr = get_pnl(sell_exit_prices, sell_prices, COMMISSION, SLIPPAGE, AVAILABLE_CAPITAL, TRADE_SIZE, 0)
    sell_pnl = np.sum(sell_arr)
    all_arr = merge_buy_sell_pnl(buy_idxs, sell_idxs, buy_arr, sell_arr)
    total_pnl = buy_pnl + sell_pnl
    equity_curve_arr = np.cumsum(all_arr)
    if len(sell_exit_idxs) == 0:
        sell_exit_idxs = np.array([-1])
        sell_arr = np.array([0])
    elif len(buy_exit_idxs) == 0:
        buy_exit_idxs[0] = np.array([-1])
        buy_arr = np.array([0])
    pnl_returns = get_returns(
        buy_idxs=buy_exit_idxs, 
        buy_pnl=buy_arr, 
        sell_idxs=sell_exit_idxs, 
        sell_pnl=sell_arr, 
        n_data=len(open_prices)
    )
    drawdowns = get_drawdowns(equity_curve_arr)
    if len(drawdowns[drawdowns!=0]) != 0:
        avg_drawdown = np.sum(drawdowns[drawdowns!=0]) / len(drawdowns[drawdowns!=0])
        fitness = total_pnl / avg_drawdown
    else:
        fitness = np.nan
        avg_drawdown = np.nan
gc.collect()'''
    
    return text_code

def create_txt_code_pnl1_vbt(buy_signal_txt, buy_exit_txt, sell_signal_txt, sell_exit_txt, 
                     fee=0.015, slippage=0.00005, inv_amount=700000, trade_size=0.5, max_lag=99):

    text_code = f'''import os
CUR_DIR = os.getcwd()
# os.chdir('src')
import pandas as pd
import numpy as np
import vectorbt as vbt
import gc
from fitness.indicators import numba_indicators_nan, signals
from fitness.performance.helper_func import merge_buy_sell_pnl, get_drawdowns, get_pnl, get_lag
from fitness.performance.helper_func import trading_signals_buy, trading_signals_sell, change_exit, get_returns_vbt
# os.chdir(CUR_DIR)
#from numba import njit
COMMISSION = {fee}
SLIPPAGE = {slippage}
AVAILABLE_CAPITAL = {inv_amount}
TRADE_SIZE = {trade_size}
MAX_LAG = {max_lag}
try:
    buy_idxs, buy_exit_idxs = trading_signals_buy(buy_signal={buy_signal_txt}, exit_signal={buy_exit_txt})
except:
    buy_idxs, buy_exit_idxs = [], []
try:
    sell_idxs, sell_exit_idxs = trading_signals_sell(sell_signal={sell_signal_txt}, exit_signal={sell_exit_txt})
except:
    sell_idxs, sell_exit_idxs = [], []
# if (len(buy_idxs) == 0 or len(buy_exit_idxs) == 0) and (len(sell_idxs) == 0 or len(sell_exit_idxs) == 0):
#     fitness = -9999999
#     avg_drawdown = -9999999
# else:
try:
    buy_idxs, buy_exit_idxs, sell_idxs, sell_exit_idxs = change_exit(buy_idxs, buy_exit_idxs, sell_idxs, sell_exit_idxs)
except:
    pass
if (len(buy_idxs) == 0 or len(buy_exit_idxs) == 0) and (len(sell_idxs) == 0 or len(sell_exit_idxs) == 0):
    fitness = np.nan
else:
    buy_entries = np.array([1 if i in buy_idxs else 0 for i in range(len(price_data['btc_open']))])
    buy_exits = np.array([1 if i in buy_exit_idxs else 0 for i in range(len(price_data['btc_open']))])
    sell_entries = np.array([1 if i in sell_idxs else 0 for i in range(len(price_data['btc_open']))])
    sell_exits = np.array([1 if i in sell_exit_idxs else 0 for i in range(len(price_data['btc_open']))])
    price_data_open = pd.Series(price_data['btc_open'].reshape(-1, ), index=pd.to_datetime(price_data['datetime']))
    pf = vbt.Portfolio.from_signals(
        price_data_open, entries=buy_entries, exits=buy_exits, 
        init_cash=AVAILABLE_CAPITAL, fees=COMMISSION, 
        slippage=SLIPPAGE, size=TRADE_SIZE, 
        short_entries=sell_entries, short_exits=sell_exits
    )
    total_return_p = pf.stats()['Total Return [%]']
    max_drawdown_p = pf.stats()['Max Drawdown [%]']
    fitness = total_return_p / max_drawdown_p
    trades = pf.trades.records
    pnl_returns = get_returns_vbt(
        idxs=trades['exit_idx'], 
        pnl=trades['pnl'],
        n_data=len(price_data['btc_open'])
    )
    all_arr = trades['pnl'].values
    equity_curve_arr = np.cumsum(all_arr)
    drawdowns = get_drawdowns(equity_curve_arr)
    if len(drawdowns[drawdowns!=0]) != 0:
        avg_drawdown = np.sum(drawdowns[drawdowns!=0]) / len(drawdowns[drawdowns!=0])
    else:
        avg_drawdown = np.nan
gc.collect()'''
    
    return text_code

def test_out_of_fold(df_str, data_path, n_fold, fold_size, time_freq, create_txt_code=create_txt_code1):

    final_entry_win_pc_df = pd.DataFrame()
    final_exit_win_pc_df = pd.DataFrame()
    final_core_win_pc_df = pd.DataFrame()
    final_perf_df = pd.DataFrame()
    final_mc_df = pd.DataFrame()

    equity_curve_dict = defaultdict(list)

    df = generate_fold_data_valid(
        data_path=data_path, 
        fold_size=fold_size,
        time_freq=time_freq,
        fold=n_fold
    )

    price_data = {}
    for col in df.columns:
        # if col == 'datetime':
        #     continue
        # else:
        price_data[col] = df[col].values.reshape(-1, 1)
    price_data['day_of_week'] = (df['datetime'].dt.dayofweek + 1).values.reshape(-1, 1)
    price_data['month'] = df['datetime'].dt.month.values.reshape(-1, 1)
    price_data['hour'] = df['datetime'].dt.hour.values.reshape(-1, 1)
    price_data['minute'] = df['datetime'].dt.minute.values.reshape(-1, 1)

    for row in tqdm(df_str.itertuples()):

        buy_signal_txt = row.buy
        buy_exit_txt = row.exit_buy
        sell_signal_txt = row.sell
        sell_exit_txt = row.exit_sell
        strategy = row.strategy

        text_code = create_txt_code(buy_signal_txt, buy_exit_txt, sell_signal_txt, sell_exit_txt)
        
        entry_test_n_not_worked = 0
        exit_test_n_not_worked = 0
        core_test_n_not_worked = 0
        perf_n_not_worked = 0
        mc_n_not_worked = 0
        n_total_cases = 1

        entry_walk_forward_dict = defaultdict(list)

        exit_walk_forward_dict = defaultdict(list)

        core_walk_forward_dict = defaultdict(list)

        performance_walk_forward_dict = defaultdict(list)

        mc_walk_forward_dict = defaultdict(list)

        exec_dict = {'price_data': price_data}
        # try:
        exec(text_code, exec_dict)
        # except Exception as e:
        #     print(f'fold = {i_fold}')
        #     print(e)
        #     print(text_code)

        try:
            equity_curve_arr = exec_dict['equity_curve_arr']
            equity_curve_dict[strategy].append(equity_curve_arr)
        except:
            pass

        try:
            fixed_winning_percent, fixed_bar_winning_percent, random_winning_percent = calculate_mean_win_perc_entry_testing(
                exec_dict, df
            )
            entry_walk_forward_dict['fixed_sp_testing'].append(fixed_winning_percent)
            entry_walk_forward_dict['fixed_bar_testing'].append(fixed_bar_winning_percent)
            entry_walk_forward_dict['random_exit_testing'].append(random_winning_percent)
        except:
            entry_test_n_not_worked += 1

        try:
            trend_winning_percent, countertrend_winning_percent, random_winning_percent = calculate_mean_win_perc_exit_testing(
                exec_dict, df
            )
            exit_walk_forward_dict['trend_entry_testing'].append(trend_winning_percent)
            exit_walk_forward_dict['countertrend_entry_testing'].append(countertrend_winning_percent)
            exit_walk_forward_dict['random_entry_testing'].append(random_winning_percent)
        except:
            exit_test_n_not_worked += 1

        try:
            winning_percent = calculate_mean_win_perc_core_testing(exec_dict)
            core_walk_forward_dict['core_testing'].append(winning_percent)
        except:
            core_test_n_not_worked += 1

        try:
            metric_dict = calculate_mean_performance(exec_dict, monkey_test=False)
            performance_walk_forward_dict['n_trades'].append(metric_dict['n_trades'])
            performance_walk_forward_dict['pnl'].append(metric_dict['overall_pnl'])
            performance_walk_forward_dict['roi'].append(metric_dict['roi'])
            performance_walk_forward_dict['avg_drawdown'].append(metric_dict['avg_drawdown'])
            performance_walk_forward_dict['drawdown'].append(metric_dict['max_dd'])
            performance_walk_forward_dict['drawdown_dur'].append(metric_dict['drawdown_dur'])
            performance_walk_forward_dict['pnl_avgd_ratio'].append(metric_dict['pnl_avgd_ratio'])
            performance_walk_forward_dict['sharpe_ratio'].append(metric_dict['sharpe_ratio'])
            performance_walk_forward_dict['sortino_ratio'].append(metric_dict['sortino_ratio'])
            if 'mt_pnl' in metric_dict.keys():
                performance_walk_forward_dict['mt_pnl'].append(metric_dict['mt_pnl'])
                performance_walk_forward_dict['mt_mdd'].append(metric_dict['mt_mdd'])
        except:
            perf_n_not_worked += 1

        try:
            mc_dict = calculate_mc_performance(exec_dict)
            mc_walk_forward_dict['median_max_dd'].append(mc_dict['median_max_dd'])
            mc_walk_forward_dict['median_dd_dur'].append(mc_dict['median_dd_dur'])
            mc_walk_forward_dict['median_profit'].append(mc_dict['median_profit'])
            mc_walk_forward_dict['median_return'].append(mc_dict['median_return'])
            mc_walk_forward_dict['return_dd_ratio'].append(mc_dict['return_dd_ratio'])
            mc_walk_forward_dict['prob_profit'].append(mc_dict['prob_profit'])
        except:
            mc_n_not_worked += 1

        gc.collect()

        temp_signal_df = pd.DataFrame(
            {
                'strategy': strategy, 
                'buy': [buy_signal_txt], 
                'buy': [buy_signal_txt], 
                'sell': [sell_signal_txt],
                'exit_buy': [buy_exit_txt], 
                'exit_sell': [sell_exit_txt]
            }
        )

        entry_win_pc_df = get_entry_win_pc_df(entry_walk_forward_dict, entry_test_n_not_worked, n_total_cases)
        entry_win_pc_df = pd.concat([temp_signal_df, entry_win_pc_df], axis=1)
        final_entry_win_pc_df = pd.concat([final_entry_win_pc_df, entry_win_pc_df])

        exit_win_pc_df = get_exit_win_pc_df(exit_walk_forward_dict, exit_test_n_not_worked, n_total_cases)
        exit_win_pc_df = pd.concat([temp_signal_df, exit_win_pc_df], axis=1)
        final_exit_win_pc_df = pd.concat([final_exit_win_pc_df, exit_win_pc_df])

        core_win_pc_df = get_core_win_pc_df(core_walk_forward_dict, core_test_n_not_worked, n_total_cases)
        core_win_pc_df = pd.concat([temp_signal_df, core_win_pc_df], axis=1)
        final_core_win_pc_df = pd.concat([final_core_win_pc_df, core_win_pc_df])

        perf_df = get_perf_df(performance_walk_forward_dict, perf_n_not_worked, n_total_cases)
        perf_df = pd.concat([temp_signal_df, perf_df], axis=1)
        final_perf_df = pd.concat([final_perf_df, perf_df])

        mc_df = get_mc_df(mc_walk_forward_dict, mc_n_not_worked, n_total_cases)
        mc_df = pd.concat([temp_signal_df, mc_df], axis=1)
        final_mc_df = pd.concat([final_mc_df, mc_df])

        # gc.collect()

    return final_entry_win_pc_df, final_exit_win_pc_df, final_core_win_pc_df, final_perf_df, final_mc_df, equity_curve_dict

def filter_save_lstr(data_path, n_fold, str_file_path, logger, fold_size, time_freq,
                     lstr_path='live_strategies', 
                     lstr_file_name='baseline', is_subset=False, start_subset=0, end_subset=10,
                     entry_testing_threshold=50, exit_testing_threshold=50, core_testing_threshold=60,
                     prob_threshold=0.8, is_counter_trend_exit=True, 
                     is_random_exit=True, entry_exit_on=True, create_txt_code=create_txt_code1):

    logger.info('Loading survived strategies from %s', str_file_path)
    try:
        df_selected_str = pd.read_csv(str_file_path)
    except:
        df_selected_str = pd.read_csv(str_file_path, sep=';')
    logger.info('Survived strategies loaded!')

    if df_selected_str.shape[0] != 0:

        logger.info('Starting out of fold testing stats calculation...')

        if is_subset:
            (final_entry_win_pc_df_fold, final_exit_win_pc_df_fold, final_core_win_pc_df_fold, 
            final_perf_df_fold, final_mc_df_fold, equity_curve_dict_fold) = test_out_of_fold(
                df_selected_str.iloc[start_subset:end_subset], data_path, n_fold,
                fold_size, time_freq, create_txt_code=create_txt_code
            )
        else:
            (final_entry_win_pc_df_fold, final_exit_win_pc_df_fold, final_core_win_pc_df_fold, 
            final_perf_df_fold, final_mc_df_fold, equity_curve_dict_fold) = test_out_of_fold(
                df_selected_str, data_path, n_fold, fold_size, time_freq, create_txt_code=create_txt_code
            )

        logger.info('Out of fold testing stats calculation completed!')

        gc.collect()

        if entry_exit_on:

            logger.info('Starting out of fold entry testing filtering...')
            entry_testing_strategies_fold = final_entry_win_pc_df_fold[
                (final_entry_win_pc_df_fold['Fixed_StopLoss_TakeProfit_testing'] >= entry_testing_threshold) & 
                (final_entry_win_pc_df_fold['Fixed_Bar_testing'] >= entry_testing_threshold) & 
                (final_entry_win_pc_df_fold['Random_Exit_testing'] >= entry_testing_threshold)
            ]['strategy'].tolist()
            logger.info('Out of fold entry testing filtering finished!')

            logger.info('Starting out of fold exit testing filtering...')
            if is_counter_trend_exit and is_random_exit:
                exit_testing_strategies_fold = final_exit_win_pc_df_fold[
                    (final_exit_win_pc_df_fold['Trend_testing'] >= exit_testing_threshold) & 
                    (final_exit_win_pc_df_fold['Countertrend_testing'] >= exit_testing_threshold) & 
                    (final_exit_win_pc_df_fold['Random_Entry_testing'] >= exit_testing_threshold) &
                    (final_exit_win_pc_df_fold['strategy'].isin(entry_testing_strategies_fold))
                ]['strategy'].tolist()
            elif is_counter_trend_exit:
                exit_testing_strategies_fold = final_exit_win_pc_df_fold[
                    (final_exit_win_pc_df_fold['Trend_testing'] >= exit_testing_threshold) & 
                    (final_exit_win_pc_df_fold['Countertrend_testing'] >= exit_testing_threshold) & 
                    (final_exit_win_pc_df_fold['strategy'].isin(entry_testing_strategies_fold))
                ]['strategy'].tolist()
            elif is_random_exit:
                exit_testing_strategies_fold = final_exit_win_pc_df_fold[
                    (final_exit_win_pc_df_fold['Trend_testing'] >= exit_testing_threshold) &  
                    (final_exit_win_pc_df_fold['Random_Entry_testing'] >= exit_testing_threshold) &
                    (final_exit_win_pc_df_fold['strategy'].isin(entry_testing_strategies_fold))
                ]['strategy'].tolist()
            else:
                exit_testing_strategies_fold = final_exit_win_pc_df_fold[
                    (final_exit_win_pc_df_fold['Trend_testing'] >= exit_testing_threshold) &  
                    (final_exit_win_pc_df_fold['strategy'].isin(entry_testing_strategies_fold))
                ]['strategy'].tolist()
            logger.info('Out of fold exit testing filtering finished!')

            logger.info('Starting out of fold core testing filtering...')
            core_testing_strategies_fold = final_core_win_pc_df_fold[
                (final_core_win_pc_df_fold['Core_Testing'] >= core_testing_threshold) &
                (final_core_win_pc_df_fold['strategy'].isin(exit_testing_strategies_fold))
            ]['strategy'].tolist()
            logger.info('Out of fold core testing filtering finished!')

        else:

            logger.info('Starting out of fold core testing filtering...')
            core_testing_strategies_fold = final_core_win_pc_df_fold[
                (final_core_win_pc_df_fold['Core_Testing'] >= core_testing_threshold)
            ]['strategy'].tolist()
            logger.info('Out of fold core testing filtering finished!')

        logger.info('Starting out of fold ROI filtering...')
        perf_strategies_fold = final_perf_df_fold[
            (final_perf_df_fold['strategy'].isin(core_testing_strategies_fold)) & 
            (final_perf_df_fold['ROI (%)'] > 0)
        ]['strategy'].tolist()
        logger.info('Out of fold ROI filtering finished!')

        logger.info('Starting out of fold Monte Carlo filtering...')
        mc_strategies_fold = final_mc_df_fold[
            (final_mc_df_fold['strategy'].isin(perf_strategies_fold)) &
            (final_mc_df_fold['prob'] > prob_threshold)
        ]['strategy'].tolist()
        logger.info('Out of fold Monte Carlo filtering finished!')

        if len(mc_strategies_fold) == 0:
            logger.info('There are not any strategies survived for live testing!')
        else:
            logger.info("Creating %s directory if it doesn't exist...", lstr_path)
            if not os.path.exists(lstr_path):
                os.mkdir(lstr_path)
            logger.info("%s directory created!", lstr_path)

            logger.info("Saving out of fold survived strategies to %s directory...", lstr_path)
            df_selected_live = df_selected_str[df_selected_str['strategy'].isin(mc_strategies_fold)]
            df_selected_live['prob'] = final_mc_df_fold[final_mc_df_fold['strategy'].isin(mc_strategies_fold)]['prob'].values.tolist()
            selected_path = f'{lstr_path}/selected_strategies_{lstr_file_name}.csv'
            df_selected_live.to_csv(selected_path, index=False)
            logger.info("Out of fold survived strategies saved!")

def creating_port_weights(lstr_path, logger, lstr_file_name='baseline', port_path='portfolio_strategies',
                          port_file_name='baseline', is_prob=False, prob_threshold=0.9):

    logger.info("Creating %s directory if it doesn't exist...", port_path)
    if not os.path.exists(port_path):
        os.mkdir(port_path)
    logger.info("%s directory created!", port_path)

    logger.info('Starting portfolio weight calculation...')

    # live_strategy_files = os.listdir(lstr_path)

    # port_df = pd.DataFrame()

    # for file in live_strategy_files:

    #     temp_df = pd.read_csv(f'{lstr_path}/{file}')
    #     # temp_df.drop(columns='strategy', inplace=True)
    #     # temp_df = temp_df['prob'].to_frame()
        
    #     port_df = pd.concat([port_df, temp_df], axis=0)

    port_df = pd.read_csv(f'{lstr_path}/selected_strategies_{lstr_file_name}.csv')
    port_df.sort_values('fitness', ascending=True, inplace=True)
    port_df.reset_index(inplace=True, drop=True)
    # port_df = port_df.iloc[:15]

    if is_prob:
        port_df = port_df[port_df['prob'] > prob_threshold]

    logger.info('Portfolio weight calculation finished!')
    
    logger.info('Saving the weights to %s directory', port_path)
    port_df.reset_index(inplace=True, drop=True)
    port_df['weight'] = port_df['prob'] / port_df['prob'].sum()
    port_df.to_csv(f'{port_path}/portfolio_{port_file_name}.csv', index=False)
    logger.info('Weights are saved to {port_path} directory!')

def creating_port_weights_mvp(lstr_path, data_path, n_fold, fold_size, time_freq, 
                              create_txt_code=create_txt_code_pnl1, freq_minutes=1,
                              lstr_file_name='baseline',
                              port_path='portfolio_strategies', port_file_name='portfolio',
                              is_min_variance_port=True, is_sharpe_port=False, is_sortino_port=False,
                              is_prob=True, prob_threshold=0.9, n_days_per_year=365):
    
    if not os.path.exists(port_path):
        os.mkdir(port_path)

    # live_strategy_files = os.listdir(lstr_path)

    # df_str = pd.DataFrame()

    # for file in live_strategy_files:

    #     temp_df = pd.read_csv(f'{lstr_path}/{file}')
    #     # temp_df.drop(columns='strategy', inplace=True)
    #     # temp_df = temp_df['prob'].to_frame()
        
    #     df_str = pd.concat([df_str, temp_df], axis=0)

    df_str = pd.read_csv(f'{lstr_path}/selected_strategies_{lstr_file_name}.csv')
    df_str.sort_values('fitness', ascending=True, inplace=True)
    df_str.reset_index(inplace=True, drop=True)
    # df_str = df_str.iloc[:15]

    if is_prob:
        df_str = df_str[df_str['prob'] > prob_threshold]

    df_str.reset_index(inplace=True, drop=True)
    
    final_port_dict = defaultdict(list)

    df = generate_fold_data(
        data_path=data_path, fold=n_fold, fold_size=fold_size, time_freq=time_freq
    )

    price_data = {}
    for col in df.columns:
        # if col == 'datetime':
        #     continue
        # else:
        price_data[col] = df[col].values.reshape(-1, 1)
    price_data['day_of_week'] = (df['datetime'].dt.dayofweek + 1).values.reshape(-1, 1)
    price_data['month'] = df['datetime'].dt.month.values.reshape(-1, 1)
    price_data['hour'] = df['datetime'].dt.hour.values.reshape(-1, 1)
    price_data['minute'] = df['datetime'].dt.minute.values.reshape(-1, 1)

    exec_dict = {'price_data': price_data}

    df_returns = pd.DataFrame()

    for row in tqdm(df_str.itertuples()):

        buy_signal_txt = row.buy
        buy_exit_txt = row.exit_buy
        sell_signal_txt = row.sell
        sell_exit_txt = row.exit_sell
        strategy = row.strategy

        text_code = create_txt_code(buy_signal_txt, buy_exit_txt, sell_signal_txt, sell_exit_txt)
        # text_code = create_txt_code1(buy_signal_txt, buy_exit_txt, sell_signal_txt, sell_exit_txt)

        try:
            exec(text_code, exec_dict)
            # print(f"fold = {i_fold}, df shape = {df.shape}, df str shape = {df_str.shape}, str = {strategy}")
            # print(len(exec_dict['buy_idxs']), len(exec_dict['buy_exit_idxs']), len(exec_dict['sell_idxs']), len(exec_dict['sell_exit_idxs']))
            # print(exec_dict['pnl_returns'])
            df_returns[strategy] = list(exec_dict['pnl_returns'])
        except:
            # print(f"str = {strategy}")
            # try:
            #     print(exec_dict['sell_arr'])
            # except:
            #     pass
            # print(e)
            continue

    str_list = list(df_returns.columns)

    df_returns['date'] = df.iloc[1:]['datetime'].dt.date.values

    # 1. Aggregate the minute log returns to daily log returns (sum them for each day)
    daily_log_returns = df_returns.groupby('date')[str_list].sum()

    # 2. Calculate the mean of daily log returns
    mean_daily_log_return = daily_log_returns.mean()

    # 3. Annualize the log return (mean daily log return * 365 days in a year)
    annualized_log_return = mean_daily_log_return * n_days_per_year  # 365 days for Bitcoin

    df_returns.drop(columns='date', inplace=True)

    annual_factor = n_days_per_year * 24 * 60 // freq_minutes

    var_matrix = df_returns.cov()

    # variance_matrix = var_matrix * annual_factor

    # port_weight = np.array([[0.5], [0.5]])

    # port_var = np.transpose(port_weight) @ variance_matrix @ port_weight
    # port_vol = np.sqrt(port_var)

    downside_dict = {}
    for col in df_returns.columns:
        temp_downside_deviations = downside_deviation(df_returns[col], target=0)
        downside_dict[col] = temp_downside_deviations

    # 2. Create a DataFrame for downside deviations
    downside_df = pd.DataFrame(downside_dict)

    # 3. Drop rows with NaN values (where there was no downside return)
    downside_df.dropna(inplace=True)

    # 4. Calculate the downside covariance matrix
    downside_cov_matrix = downside_df.cov()

    # print(df_returns)
    # print(annualized_log_return.shape)
    # print(var_matrix.shape)
    # print(downside_cov_matrix.shape)
    # print(annual_factor)

    port_returns, port_volatility, port_weights, port_downside_volatility = calculate_portfolio_frontier(
        df_returns, annualized_log_return, var_matrix, downside_cov_matrix, annual_factor)
    
    data_dict = {'Returns': port_returns, 'Volatility': port_volatility, 'Downside_Volatility': port_downside_volatility}

    for idx, str_name in enumerate(str_list):
        data_dict[str_name] = [elem[idx] for elem in port_weights]

    port_df = pd.DataFrame(data_dict)

    if is_min_variance_port:
        try:
            temp_port_dict = port_df.iloc[port_df['Volatility'].idxmin()][list(df_returns.columns)].to_dict()
        except:
            temp_port_dict = {}
    elif is_sharpe_port:
        try:
            temp_port_dict = port_df.iloc[(port_df['Returns'] / port_df['Volatility']).idxmax()][list(df_returns.columns)].to_dict()
        except:
            temp_port_dict = {}
    elif is_sortino_port:
        try:
            temp_port_dict = port_df.iloc[(port_df['Returns'] / port_df['Downside_Volatility']).idxmax()][list(df_returns.columns)].to_dict()
        except:
            temp_port_dict = {}

    if len(temp_port_dict) == 0:
        for k in temp_port_dict.keys():
            final_port_dict[k].append(np.nan)
    else:
        for k in temp_port_dict.keys():
            final_port_dict[k].append(temp_port_dict[k])

    gc.collect()

    if is_min_variance_port:
        port_file_name += '_mvp'
    elif is_sharpe_port:
        port_file_name += '_sharpe'
    elif is_sortino_port:
        port_file_name += '_sortino'

    avg_weight_sum = np.sum([np.nanmean(final_port_dict[k]) for k in final_port_dict.keys()])

    port_map_dict = {k: np.nanmean(final_port_dict[k]) / avg_weight_sum for k in final_port_dict.keys()}

    df_str['weight'] = df_str['strategy'].map(port_map_dict)
    df_str['weight'] = df_str['weight'].apply(lambda x: 0 if pd.isna(x) else x)
    df_str.to_csv(f'{port_path}/{port_file_name}.csv', index=False)

def creating_port_weights_hrp(lstr_path, data_path, n_fold, fold_size, time_freq, 
                              create_txt_code=create_txt_code_pnl1,
                              lstr_file_name='baseline',
                              port_path='portfolio_strategies', port_file_name='portfolio_hrp',
                              is_prob=True, prob_threshold=0.9):
    
    if not os.path.exists(port_path):
        os.mkdir(port_path)

    # live_strategy_files = os.listdir(lstr_path)

    # df_str = pd.DataFrame()

    # for file in live_strategy_files:

    #     temp_df = pd.read_csv(f'{lstr_path}/{file}')
    #     # temp_df.drop(columns='strategy', inplace=True)
    #     # temp_df = temp_df['prob'].to_frame()
        
    #     df_str = pd.concat([df_str, temp_df], axis=0)

    df_str = pd.read_csv(f'{lstr_path}/selected_strategies_{lstr_file_name}.csv')
    df_str.sort_values('fitness', ascending=True, inplace=True)
    df_str.reset_index(inplace=True, drop=True)
    # df_str = df_str.iloc[:15]

    if is_prob:
        df_str = df_str[df_str['prob'] > prob_threshold]

    df_str.reset_index(inplace=True, drop=True)
    
    final_port_dict = defaultdict(list)

    df = generate_fold_data(
        data_path=data_path, fold=n_fold, fold_size=fold_size, time_freq=time_freq
    )

    price_data = {}
    for col in df.columns:
        # if col == 'datetime':
        #     continue
        # else:
        price_data[col] = df[col].values.reshape(-1, 1)
    price_data['day_of_week'] = (df['datetime'].dt.dayofweek + 1).values.reshape(-1, 1)
    price_data['month'] = df['datetime'].dt.month.values.reshape(-1, 1)
    price_data['hour'] = df['datetime'].dt.hour.values.reshape(-1, 1)
    price_data['minute'] = df['datetime'].dt.minute.values.reshape(-1, 1)

    exec_dict = {'price_data': price_data}

    df_returns = pd.DataFrame()

    for row in tqdm(df_str.itertuples()):

        buy_signal_txt = row.buy
        buy_exit_txt = row.exit_buy
        sell_signal_txt = row.sell
        sell_exit_txt = row.exit_sell
        strategy = row.strategy

        text_code = create_txt_code(buy_signal_txt, buy_exit_txt, sell_signal_txt, sell_exit_txt)

        try:
            exec(text_code, exec_dict)
            df_returns[strategy] = list(exec_dict['pnl_returns'])
        except:
            continue
    
    try:
        correlation_matrix = df_returns.corr()

        distance_matrix = 1 - correlation_matrix

        linkage_matrix = linkage(squareform(distance_matrix), method='ward')

        clusters = fcluster(linkage_matrix, t=1.5, criterion='distance')

        covariance_matrix = df_returns.cov()

        cluster_weights = {}
        for cluster_id in np.unique(clusters):
            cluster_strategies = [df_returns.columns[i] for i in range(len(clusters)) if clusters[i] == cluster_id]
            cluster_cov = covariance_matrix.loc[cluster_strategies, cluster_strategies]
            cluster_weights[cluster_id] = inverse_variance_weights(cluster_cov)

        # cluster_sizes = {cid: sum(clusters == cid) for cid in np.unique(clusters)}
        total_cluster_weights = {cid: np.sum(cluster_weights[cid]) / len(cluster_weights) for cid in cluster_weights}
        final_weights = {}

        for cid, strategies in cluster_weights.items():
            for strategy, weight in zip(df_returns.columns[clusters == cid], strategies):
                final_weights[strategy] = weight * total_cluster_weights[cid]

        final_weights = {k: v / sum(final_weights.values()) for k, v in final_weights.items()}

        for k in final_weights.keys():
            final_port_dict[k].append(final_weights[k])

    except:
        pass

    gc.collect()

    avg_weight_sum = np.sum([np.nanmean(final_port_dict[k]) for k in final_port_dict.keys()])

    port_map_dict = {k: np.nanmean(final_port_dict[k]) / avg_weight_sum for k in final_port_dict.keys()}

    df_str['weight'] = df_str['strategy'].map(port_map_dict)
    df_str['weight'] = df_str['weight'].apply(lambda x: 0 if pd.isna(x) else x)
    df_str.to_csv(f'{port_path}/{port_file_name}.csv', index=False)

def creating_port_weights_kmeans(lstr_path, data_path, n_fold, fold_size, time_freq, 
                              create_txt_code=create_txt_code_pnl1,
                              lstr_file_name='baseline',
                              port_path='portfolio_strategies', 
                              port_file_name='portfolio',
                              is_prob=True, 
                              prob_threshold=0.9,
                              select_method='min-dist'
):
    
    if not os.path.exists(port_path):
        os.mkdir(port_path)

    # live_strategy_files = os.listdir(lstr_path)

    # df_str = pd.DataFrame()

    # for file in live_strategy_files:

    #     temp_df = pd.read_csv(f'{lstr_path}/{file}')
    #     # temp_df.drop(columns='strategy', inplace=True)
    #     # temp_df = temp_df['prob'].to_frame()
        
    #     df_str = pd.concat([df_str, temp_df], axis=0)

    df_str = pd.read_csv(f'{lstr_path}/selected_strategies_{lstr_file_name}.csv')

    if is_prob:
        df_str = df_str[df_str['prob'] > prob_threshold]

    df_str.reset_index(inplace=True, drop=True)

    if df_str.shape[0] < 30:
        return

    df = generate_fold_data(
        data_path=data_path, fold=n_fold, fold_size=fold_size, time_freq=time_freq
    )

    price_data = {}
    for col in df.columns:
        # if col == 'datetime':
        #     continue
        # else:
        price_data[col] = df[col].values.reshape(-1, 1)
    price_data['day_of_week'] = (df['datetime'].dt.dayofweek + 1).values.reshape(-1, 1)
    price_data['month'] = df['datetime'].dt.month.values.reshape(-1, 1)
    price_data['hour'] = df['datetime'].dt.hour.values.reshape(-1, 1)
    price_data['minute'] = df['datetime'].dt.minute.values.reshape(-1, 1)

    exec_dict = {'price_data': price_data}

    list_mean = []

    list_std = []

    strategy_list = []

    for row in tqdm(df_str.itertuples()):

        buy_signal_txt = row.buy
        buy_exit_txt = row.exit_buy
        sell_signal_txt = row.sell
        sell_exit_txt = row.exit_sell
        strategy = row.strategy

        text_code = create_txt_code(buy_signal_txt, buy_exit_txt, sell_signal_txt, sell_exit_txt)
        # text_code = create_txt_code1(buy_signal_txt, buy_exit_txt, sell_signal_txt, sell_exit_txt)

        try:
            exec(text_code, exec_dict)
            # print(f"fold = {i_fold}, df shape = {df.shape}, df str shape = {df_str.shape}, str = {strategy}")
            # print(len(exec_dict['buy_idxs']), len(exec_dict['buy_exit_idxs']), len(exec_dict['sell_idxs']), len(exec_dict['sell_exit_idxs']))
            # print(exec_dict['pnl_returns'])
            strategy_list.append(strategy)
            list_mean.append(np.mean(exec_dict['pnl_returns']))
            list_std.append(np.mean(exec_dict['pnl_returns']))
        except:
            # print(f"str = {strategy}")
            # try:
            #     print(exec_dict['sell_arr'])
            # except:
            #     pass
            # print(e)
            continue

    if len(list_mean) == 0:
        selected_ccstr_list = [strategy]
        df_str = df_str[df_str['strategy'].isin(selected_ccstr_list)]
        df_str['weight'] = list(np.zeros(len(selected_ccstr_list)) / len(selected_ccstr_list))
        df_str.to_csv(f'{port_path}/{port_file_name}.csv', index=False)
        return

    df_returns = pd.DataFrame()
    df_returns['strategy'] = strategy_list
    df_returns['mean'] = list_mean
    df_returns['std'] = list_std
    df_returns.dropna(inplace=True)
    df_returns.reset_index(drop=True, inplace=True)

    data = df_returns[['mean', 'std']]

    silhouette_avg = []
    for num_clusters in list(range(2,10)):
        kmeans = KMeans(n_clusters=num_clusters, init = "k-means++", n_init = 10)
        kmeans.fit_predict(data)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_avg.append(score)

    best_n_clusters = np.argmax(silhouette_avg) + 2
    kmeans = KMeans(n_clusters=best_n_clusters, init = "k-means++", n_init = 10)
    kmeans.fit_predict(data)

    df_returns['clusters'] = kmeans.labels_

    selected_ccstr_list = []

    if select_method == 'min_dist':

        mean_map_dict = df_returns.groupby('clusters')[['mean', 'std']].mean().to_dict()
        df_returns['mean_m'] = df_returns['clusters'].map(mean_map_dict['mean'])
        df_returns['std_m'] = df_returns['clusters'].map(mean_map_dict['std'])
        df_returns['dist'] = np.sqrt((df_returns['mean'] - df_returns['mean_m']) ** 2 + (df_returns['std'] - df_returns['std_m']) ** 2)


        for n_cluster in list(df_returns['clusters'].unique()):
            selected_ccstr_list.append(
                df_returns[df_returns['clusters'] == n_cluster].sort_values('dist', ascending=True).iloc[0]['strategy']
            )

    elif select_method == 'min_var':

        for n_cluster in list(df_returns['clusters'].unique()):
            selected_ccstr_list.append(
                df_returns[df_returns['clusters'] == n_cluster].sort_values('std', ascending=True).iloc[0]['strategy']
            )

    elif select_method == 'max_return':

        for n_cluster in list(df_returns['clusters'].unique()):
            selected_ccstr_list.append(
                df_returns[df_returns['clusters'] == n_cluster].sort_values('mean', ascending=False).iloc[0]['strategy']
            )

    df_str = df_str[df_str['strategy'].isin(selected_ccstr_list)]
    df_str['weight'] = list(np.ones(len(selected_ccstr_list)) / len(selected_ccstr_list))
    df_str.to_csv(f'{port_path}/{port_file_name}.csv', index=False)

def creating_port_weights_equal(
    lstr_path,
    lstr_file_name='baseline',
    port_path='portfolio_strategies', 
    port_file_name='portfolio',
):
    
    if not os.path.exists(port_path):
        os.mkdir(port_path)

    # live_strategy_files = os.listdir(lstr_path)

    # df_str = pd.DataFrame()

    # for file in live_strategy_files:

    #     temp_df = pd.read_csv(f'{lstr_path}/{file}')
    #     # temp_df.drop(columns='strategy', inplace=True)
    #     # temp_df = temp_df['prob'].to_frame()
        
    #     df_str = pd.concat([df_str, temp_df], axis=0)

    df_str = pd.read_csv(f'{lstr_path}/selected_strategies_{lstr_file_name}.csv')
    df_str['weight'] = list(np.ones(df_str.shape[0]) / df_str.shape[0])
    df_str.to_csv(f'{port_path}/{port_file_name}.csv', index=False)

def calculate_port_stats(df_port, df, create_txt_code_port=create_txt_code_port1):

    # final_entry_win_pc_df = pd.DataFrame()
    # final_exit_win_pc_df = pd.DataFrame()
    # final_core_win_pc_df = pd.DataFrame()
    final_perf_df = pd.DataFrame()
    # final_mc_df = pd.DataFrame()

    equity_curve_dict = defaultdict(list)

    # strategy_idx = 1

    for row in tqdm(df_port.itertuples()):

        buy_signal_txt = row.buy
        buy_exit_txt = row.exit_buy
        sell_signal_txt = row.sell
        sell_exit_txt = row.exit_sell
        weight = row.weight
        strategy_name = row.strategy

        text_code = create_txt_code_port(buy_signal_txt, buy_exit_txt, sell_signal_txt, sell_exit_txt, weight)
        
        # entry_test_n_not_worked = 0
        # exit_test_n_not_worked = 0
        # core_test_n_not_worked = 0
        perf_n_not_worked = 0
        # mc_n_not_worked = 0
        n_total_cases = 0

        # entry_walk_forward_dict = defaultdict(list)

        # exit_walk_forward_dict = defaultdict(list)

        # core_walk_forward_dict = defaultdict(list)

        performance_walk_forward_dict = defaultdict(list)

        # mc_walk_forward_dict = defaultdict(list)

        n_total_cases += 1

        # df = df_52w.iloc[idx:idx+bars_per_5week, :]
        # df.reset_index(drop=True, inplace=True)
        # df = generate_fold_data(data_path, fold=n_fold)
        # df.reset_index(drop=True, inplace=True)

        price_data = {}
        for col in df.columns:
            # if col == 'datetime':
            #     continue
            # else:
            price_data[col] = df[col].values.reshape(-1, 1)
        price_data['day_of_week'] = (df['datetime'].dt.dayofweek + 1).values.reshape(-1, 1)
        price_data['month'] = df['datetime'].dt.month.values.reshape(-1, 1)
        price_data['hour'] = df['datetime'].dt.hour.values.reshape(-1, 1)
        price_data['minute'] = df['datetime'].dt.minute.values.reshape(-1, 1)

        exec_dict = {'price_data': price_data}
        # try:
        exec(text_code, exec_dict)
        # except:
        #     pass

        try:
            equity_curve_arr = exec_dict['equity_curve_arr']
            equity_curve_dict[strategy_name].append(equity_curve_arr)
            # print(f"fold{n_fold}: {len(equity_curve_arr)}")
        except:
            pass

        # try:
        #     fixed_winning_percent, fixed_bar_winning_percent, random_winning_percent = calculate_mean_win_perc_entry_testing(exec_dict, df)
        #     entry_walk_forward_dict['fixed_sp_testing'].append(fixed_winning_percent)
        #     entry_walk_forward_dict['fixed_bar_testing'].append(fixed_bar_winning_percent)
        #     entry_walk_forward_dict['random_exit_testing'].append(random_winning_percent)
        # except:
        #     entry_test_n_not_worked += 1

        # try:
        #     trend_winning_percent, countertrend_winning_percent, random_winning_percent = calculate_mean_win_perc_exit_testing(exec_dict, df)
        #     exit_walk_forward_dict['trend_entry_testing'].append(trend_winning_percent)
        #     exit_walk_forward_dict['countertrend_entry_testing'].append(countertrend_winning_percent)
        #     exit_walk_forward_dict['random_entry_testing'].append(random_winning_percent)
        # except:
        #     exit_test_n_not_worked += 1

        # try:
        #     winning_percent = calculate_mean_win_perc_core_testing(exec_dict)
        #     core_walk_forward_dict['core_testing'].append(winning_percent)
        # except:
        #     core_test_n_not_worked += 1

        try:
            metric_dict = calculate_mean_performance(exec_dict, monkey_test=False)
            performance_walk_forward_dict['n_trades'].append(metric_dict['n_trades'])
            performance_walk_forward_dict['pnl'].append(metric_dict['overall_pnl'])
            performance_walk_forward_dict['roi'].append(metric_dict['roi'])
            performance_walk_forward_dict['avg_drawdown'].append(metric_dict['avg_drawdown'])
            performance_walk_forward_dict['drawdown'].append(metric_dict['max_dd'])
            performance_walk_forward_dict['drawdown_dur'].append(metric_dict['drawdown_dur'])
            performance_walk_forward_dict['pnl_avgd_ratio'].append(metric_dict['pnl_avgd_ratio'])
            performance_walk_forward_dict['sharpe_ratio'].append(metric_dict['sharpe_ratio'])
            performance_walk_forward_dict['sortino_ratio'].append(metric_dict['sortino_ratio'])
            if 'mt_pnl' in metric_dict.keys():
                performance_walk_forward_dict['mt_pnl'].append(metric_dict['mt_pnl'])
                performance_walk_forward_dict['mt_mdd'].append(metric_dict['mt_mdd'])
        except:
            perf_n_not_worked += 1

        # try:
        #     mc_dict = calculate_mc_performance(exec_dict)
        #     mc_walk_forward_dict['median_max_dd'].append(mc_dict['median_max_dd'])
        #     mc_walk_forward_dict['median_dd_dur'].append(mc_dict['median_dd_dur'])
        #     mc_walk_forward_dict['median_profit'].append(mc_dict['median_profit'])
        #     mc_walk_forward_dict['median_return'].append(mc_dict['median_return'])
        #     mc_walk_forward_dict['return_dd_ratio'].append(mc_dict['return_dd_ratio'])
        #     mc_walk_forward_dict['prob_profit'].append(mc_dict['prob_profit'])
        # except:
        #     mc_n_not_worked += 1

        # temp_signal_df = pd.DataFrame({'strategy': f'strategy{strategy_idx}', 'buy': [buy_signal_txt], 'sell': [sell_signal_txt]})

        temp_signal_df = pd.DataFrame(
            {
                'strategy': strategy_name, #f'strategy{strategy_idx}', 
                'buy': [buy_signal_txt], 
                'sell': [sell_signal_txt],
                'exit_buy': [buy_exit_txt], 
                'exit_sell': [sell_exit_txt]
            }
        )

        # entry_win_pc_df = get_entry_win_pc_df(entry_walk_forward_dict, entry_test_n_not_worked, n_total_cases)
        # entry_win_pc_df = pd.concat([temp_signal_df, entry_win_pc_df], axis=1)
        # final_entry_win_pc_df = pd.concat([final_entry_win_pc_df, entry_win_pc_df])

        # exit_win_pc_df = get_exit_win_pc_df(exit_walk_forward_dict, exit_test_n_not_worked, n_total_cases)
        # exit_win_pc_df = pd.concat([temp_signal_df, exit_win_pc_df], axis=1)
        # final_exit_win_pc_df = pd.concat([final_exit_win_pc_df, exit_win_pc_df])

        # core_win_pc_df = get_core_win_pc_df(core_walk_forward_dict, core_test_n_not_worked, n_total_cases)
        # core_win_pc_df = pd.concat([temp_signal_df, core_win_pc_df], axis=1)
        # final_core_win_pc_df = pd.concat([final_core_win_pc_df, core_win_pc_df])

        perf_df = get_perf_df(performance_walk_forward_dict, perf_n_not_worked, n_total_cases)
        perf_df = pd.concat([temp_signal_df, perf_df], axis=1)
        final_perf_df = pd.concat([final_perf_df, perf_df])

        # mc_df = get_mc_df(mc_walk_forward_dict, mc_n_not_worked, n_total_cases)
        # mc_df = pd.concat([temp_signal_df, mc_df], axis=1)
        # final_mc_df = pd.concat([final_mc_df, mc_df])

        # strategy_idx += 1

        gc.collect()

    return final_perf_df, equity_curve_dict

def calculate_port_out_sample_perf(data_path, port_file_path, logger, 
                                   n_fold, fold_size, time_freq,
                                   initial_amount=350000,  
                                   port_perf_path='portfolio_out_sample_performance',
                                   port_perf_file_name='baseline', create_txt_code_port=create_txt_code_port1):
    
    logger.info(f"Creating {port_perf_path} directory if it doesn't exist...")
    if not os.path.exists(port_perf_path):
        os.mkdir(port_perf_path)
    logger.info(f"{port_perf_path} directory created!")

    try:
        logger.info(f"Loading portfolio weights from {port_file_path}...")
        port_df = pd.read_csv(port_file_path)
        port_df.dropna(inplace=True)
        port_df.reset_index(drop=True, inplace=True)
        logger.info('Weights loaded!')
    except:
        return

    logger.info('Starting out of sample ROI calculation...')

    roi_list = []
    pnl_list = []
    fold_list = []
    # nt_list = []
    # md_list = []
    # sr_list = []

    logger.info(f"Starting portfolio ROI calculation for validation sample...")

    df_prices = generate_fold_data_valid(
        data_path=data_path, fold_size=fold_size, time_freq=time_freq, fold=n_fold
    )

    final_perf_df_port, equity_curve_dict_port = calculate_port_stats(
        df_port=port_df.copy(), df=df_prices.copy(), create_txt_code_port=create_txt_code_port
    )

    port_roi = 100 * final_perf_df_port['PNL'].sum() / initial_amount

    # print(f'fold{i} ROI: {port_roi:.4f} %')

    roi_list.append(port_roi)
    fold_list.append('validation')
    pnl_list.append(final_perf_df_port['PNL'].sum())
    # nt_list.append(final_perf_df_port['N_Trades'])
    # md_list.append(final_perf_df_port['Drawdown (%)'])
    # sr_list.append(final_perf_df_port['Sharpe_Ratio'])

    logger.info(f"portfolio ROI for validation sample: {str(port_roi)} %")

    logger.info(f"Saving the portfolio performance to {port_perf_path} directory...")
    port_perf_df = pd.DataFrame()
    port_perf_df['fold'] = fold_list
    port_perf_df['ROI (%)'] = roi_list
    port_perf_df['PNL'] = pnl_list
    # port_perf_df['N_Trades'] = nt_list
    # port_perf_df['Drawdown (%)'] = md_list
    # port_perf_df['Sharpe_Ratio'] = sr_list
    port_perf_df.to_csv(f'{port_perf_path}/perf_{port_perf_file_name}.csv', index=False)
    logger.info('Portfolio performance saved!')

def calculate_port_test_sample_perf(data_path, port_file_path, logger, 
                                   n_fold, fold_size, time_freq,
                                   initial_amount=350000,  
                                   port_perf_path='portfolio_test_sample_performance',
                                   port_perf_file_name='baseline', create_txt_code_port=create_txt_code_port1):
    
    logger.info(f"Creating {port_perf_path} directory if it doesn't exist...")
    if not os.path.exists(port_perf_path):
        os.mkdir(port_perf_path)
    logger.info(f"{port_perf_path} directory created!")

    try:
        logger.info(f"Loading portfolio weights from {port_file_path}...")
        port_df = pd.read_csv(port_file_path)
        port_df.dropna(inplace=True)
        port_df.reset_index(drop=True, inplace=True)
        logger.info('Weights loaded!')
    except:
        return

    logger.info('Starting out of sample ROI calculation...')

    roi_list = []
    pnl_list = []
    fold_list = []

    logger.info(f"Starting portfolio ROI calculation for test sample...")

    df_prices = generate_fold_data_test(
        data_path=data_path, fold_size=fold_size, time_freq=time_freq, fold=n_fold
    )

    final_perf_df_port, equity_curve_dict_port = calculate_port_stats(
        df_port=port_df.copy(), df=df_prices.copy(), create_txt_code_port=create_txt_code_port
    )

    port_roi = 100 * final_perf_df_port['PNL'].sum() / initial_amount

    # print(f'fold{i} ROI: {port_roi:.4f} %')

    roi_list.append(port_roi)
    fold_list.append('test')
    pnl_list.append(final_perf_df_port['PNL'].sum())

    logger.info(f"portfolio ROI for test sample: {str(port_roi)} %")

    logger.info(f"Saving the portfolio performance to {port_perf_path} directory...")
    port_perf_df = pd.DataFrame()
    port_perf_df['fold'] = fold_list
    port_perf_df['ROI (%)'] = roi_list
    port_perf_df['PNL'] = pnl_list
    port_perf_df.to_csv(f'{port_perf_path}/perf_{port_perf_file_name}.csv', index=False)
    logger.info('Portfolio performance saved!')

def calculate_str_stats(df_str, df, create_txt_code_port=create_txt_code1):

    # final_entry_win_pc_df = pd.DataFrame()
    # final_exit_win_pc_df = pd.DataFrame()
    # final_core_win_pc_df = pd.DataFrame()
    final_perf_df = pd.DataFrame()
    # final_mc_df = pd.DataFrame()

    equity_curve_dict = defaultdict(list)

    # strategy_idx = 1

    for row in tqdm(df_str.itertuples()):

        buy_signal_txt = row.buy
        buy_exit_txt = row.exit_buy
        sell_signal_txt = row.sell
        sell_exit_txt = row.exit_sell
        strategy_name = row.strategy

        text_code = create_txt_code_port(buy_signal_txt, buy_exit_txt, sell_signal_txt, sell_exit_txt)
        
        # entry_test_n_not_worked = 0
        # exit_test_n_not_worked = 0
        # core_test_n_not_worked = 0
        perf_n_not_worked = 0
        # mc_n_not_worked = 0
        n_total_cases = 0

        # entry_walk_forward_dict = defaultdict(list)

        # exit_walk_forward_dict = defaultdict(list)

        # core_walk_forward_dict = defaultdict(list)

        performance_walk_forward_dict = defaultdict(list)

        # mc_walk_forward_dict = defaultdict(list)

        n_total_cases += 1

        # df = df_52w.iloc[idx:idx+bars_per_5week, :]
        # df.reset_index(drop=True, inplace=True)
        # df = generate_fold_data(data_path, fold=n_fold)
        # df.reset_index(drop=True, inplace=True)

        price_data = {}
        for col in df.columns:
            # if col == 'datetime':
            #     continue
            # else:
            price_data[col] = df[col].values.reshape(-1, 1)
        price_data['day_of_week'] = (df['datetime'].dt.dayofweek + 1).values.reshape(-1, 1)
        price_data['month'] = df['datetime'].dt.month.values.reshape(-1, 1)
        price_data['hour'] = df['datetime'].dt.hour.values.reshape(-1, 1)
        price_data['minute'] = df['datetime'].dt.minute.values.reshape(-1, 1)

        exec_dict = {'price_data': price_data}
        # try:
        exec(text_code, exec_dict)
        # except Exception as e:
        #     print(e)
        #     print(text_code)
        #     pass

        try:
            equity_curve_arr = exec_dict['equity_curve_arr']
            equity_curve_dict[strategy_name].append(equity_curve_arr)
            # print(f"fold{n_fold}: {len(equity_curve_arr)}")
        except:
            pass

        # try:
        #     fixed_winning_percent, fixed_bar_winning_percent, random_winning_percent = calculate_mean_win_perc_entry_testing(exec_dict, df)
        #     entry_walk_forward_dict['fixed_sp_testing'].append(fixed_winning_percent)
        #     entry_walk_forward_dict['fixed_bar_testing'].append(fixed_bar_winning_percent)
        #     entry_walk_forward_dict['random_exit_testing'].append(random_winning_percent)
        # except:
        #     entry_test_n_not_worked += 1

        # try:
        #     trend_winning_percent, countertrend_winning_percent, random_winning_percent = calculate_mean_win_perc_exit_testing(exec_dict, df)
        #     exit_walk_forward_dict['trend_entry_testing'].append(trend_winning_percent)
        #     exit_walk_forward_dict['countertrend_entry_testing'].append(countertrend_winning_percent)
        #     exit_walk_forward_dict['random_entry_testing'].append(random_winning_percent)
        # except:
        #     exit_test_n_not_worked += 1

        # try:
        #     winning_percent = calculate_mean_win_perc_core_testing(exec_dict)
        #     core_walk_forward_dict['core_testing'].append(winning_percent)
        # except:
        #     core_test_n_not_worked += 1

        try:
            metric_dict = calculate_mean_performance(exec_dict, monkey_test=False)
            performance_walk_forward_dict['n_trades'].append(metric_dict['n_trades'])
            performance_walk_forward_dict['pnl'].append(metric_dict['overall_pnl'])
            performance_walk_forward_dict['roi'].append(metric_dict['roi'])
            performance_walk_forward_dict['avg_drawdown'].append(metric_dict['avg_drawdown'])
            performance_walk_forward_dict['drawdown'].append(metric_dict['max_dd'])
            performance_walk_forward_dict['drawdown_dur'].append(metric_dict['drawdown_dur'])
            performance_walk_forward_dict['pnl_avgd_ratio'].append(metric_dict['pnl_avgd_ratio'])
            performance_walk_forward_dict['sharpe_ratio'].append(metric_dict['sharpe_ratio'])
            performance_walk_forward_dict['sortino_ratio'].append(metric_dict['sortino_ratio'])
            if 'mt_pnl' in metric_dict.keys():
                performance_walk_forward_dict['mt_pnl'].append(metric_dict['mt_pnl'])
                performance_walk_forward_dict['mt_mdd'].append(metric_dict['mt_mdd'])
        except:
            perf_n_not_worked += 1

        # try:
        #     mc_dict = calculate_mc_performance(exec_dict)
        #     mc_walk_forward_dict['median_max_dd'].append(mc_dict['median_max_dd'])
        #     mc_walk_forward_dict['median_dd_dur'].append(mc_dict['median_dd_dur'])
        #     mc_walk_forward_dict['median_profit'].append(mc_dict['median_profit'])
        #     mc_walk_forward_dict['median_return'].append(mc_dict['median_return'])
        #     mc_walk_forward_dict['return_dd_ratio'].append(mc_dict['return_dd_ratio'])
        #     mc_walk_forward_dict['prob_profit'].append(mc_dict['prob_profit'])
        # except:
        #     mc_n_not_worked += 1

        # temp_signal_df = pd.DataFrame({'strategy': f'strategy{strategy_idx}', 'buy': [buy_signal_txt], 'sell': [sell_signal_txt]})

        temp_signal_df = pd.DataFrame(
            {
                'strategy': strategy_name, #f'strategy{strategy_idx}', 
                'buy': [buy_signal_txt], 
                'sell': [sell_signal_txt],
                'exit_buy': [buy_exit_txt], 
                'exit_sell': [sell_exit_txt]
            }
        )

        # entry_win_pc_df = get_entry_win_pc_df(entry_walk_forward_dict, entry_test_n_not_worked, n_total_cases)
        # entry_win_pc_df = pd.concat([temp_signal_df, entry_win_pc_df], axis=1)
        # final_entry_win_pc_df = pd.concat([final_entry_win_pc_df, entry_win_pc_df])

        # exit_win_pc_df = get_exit_win_pc_df(exit_walk_forward_dict, exit_test_n_not_worked, n_total_cases)
        # exit_win_pc_df = pd.concat([temp_signal_df, exit_win_pc_df], axis=1)
        # final_exit_win_pc_df = pd.concat([final_exit_win_pc_df, exit_win_pc_df])

        # core_win_pc_df = get_core_win_pc_df(core_walk_forward_dict, core_test_n_not_worked, n_total_cases)
        # core_win_pc_df = pd.concat([temp_signal_df, core_win_pc_df], axis=1)
        # final_core_win_pc_df = pd.concat([final_core_win_pc_df, core_win_pc_df])

        perf_df = get_perf_df(performance_walk_forward_dict, perf_n_not_worked, n_total_cases)
        perf_df = pd.concat([temp_signal_df, perf_df], axis=1)
        final_perf_df = pd.concat([final_perf_df, perf_df])

        # mc_df = get_mc_df(mc_walk_forward_dict, mc_n_not_worked, n_total_cases)
        # mc_df = pd.concat([temp_signal_df, mc_df], axis=1)
        # final_mc_df = pd.concat([final_mc_df, mc_df])

        # strategy_idx += 1

        gc.collect()

    return final_perf_df, equity_curve_dict

def calculate_str_out_sample_perf(data_path, port_file_path, logger,
    n_fold, fold_size, time_freq,           
    initial_amount=350000, 
    str_perf_path='str_out_sample_performance',
    str_perf_file_name='baseline',
    create_txt_code_port=create_txt_code1
):
    
    logger.info(f"Creating {str_perf_path} directory if it doesn't exist...")
    if not os.path.exists(str_perf_path):
        os.mkdir(str_perf_path)
    logger.info(f"{str_perf_path} directory created!")

    logger.info(f"Loading portfolio strategies from {port_file_path}...")
    port_df = pd.read_csv(port_file_path)
    port_df.dropna(inplace=True)
    port_df.reset_index(drop=True, inplace=True)
    logger.info('Strategies loaded!')

    logger.info('Starting out of sample ROI calculation...')

    logger.info(f"Starting portfolio ROI calculation for validation sample...")

    df_prices = generate_fold_data_valid(
        data_path=data_path, fold=n_fold, fold_size=fold_size, time_freq=time_freq
    )

    temp_perf_df, equity_curve_dict_port = calculate_str_stats(
        df_str=port_df.copy(), df=df_prices.copy(), create_txt_code_port=create_txt_code_port
    )

    temp_perf_df['ROI'] = 100 * temp_perf_df['PNL'] / initial_amount

    temp_perf_df = temp_perf_df[['strategy', 'ROI']]
    temp_perf_df.rename(columns={'ROI': 'validation'}, inplace=True)

    # if i == start_fold:
    #     str_perf_df = pd.concat([str_perf_df, temp_perf_df])
    # else:
    #     str_perf_df = pd.merge(str_perf_df, temp_perf_df, on='strategy')

    logger.info(f"Saving the strategies' performance to {str_perf_path} directory...")
    temp_perf_df.to_csv(f'{str_perf_path}/perf_{str_perf_file_name}.csv', index=False)
    logger.info("Strategies' performance saved!")

def calculate_str_test_sample_perf(data_path, port_file_path, logger,
    n_fold, fold_size, time_freq,           
    initial_amount=350000, 
    str_perf_path='str_test_sample_performance',
    str_perf_file_name='baseline',
    create_txt_code_port=create_txt_code1
):
    
    logger.info(f"Creating {str_perf_path} directory if it doesn't exist...")
    if not os.path.exists(str_perf_path):
        os.mkdir(str_perf_path)
    logger.info(f"{str_perf_path} directory created!")

    logger.info(f"Loading portfolio strategies from {port_file_path}...")
    port_df = pd.read_csv(port_file_path)
    port_df.dropna(inplace=True)
    port_df.reset_index(drop=True, inplace=True)
    logger.info('Strategies loaded!')

    logger.info('Starting out of sample ROI calculation...')

    logger.info(f"Starting portfolio ROI calculation for test sample...")

    df_prices = generate_fold_data_test(
        data_path=data_path, fold=n_fold, fold_size=fold_size, time_freq=time_freq
    )

    temp_perf_df, equity_curve_dict_port = calculate_str_stats(
        df_str=port_df.copy(), df=df_prices.copy(), create_txt_code_port=create_txt_code_port
    )

    temp_perf_df['ROI'] = 100 * temp_perf_df['PNL'] / initial_amount

    temp_perf_df = temp_perf_df[['strategy', 'ROI']]
    temp_perf_df.rename(columns={'ROI': 'test'}, inplace=True)

    # if i == start_fold:
    #     str_perf_df = pd.concat([str_perf_df, temp_perf_df])
    # else:
    #     str_perf_df = pd.merge(str_perf_df, temp_perf_df, on='strategy')

    logger.info(f"Saving the strategies' performance to {str_perf_path} directory...")
    temp_perf_df.to_csv(f'{str_perf_path}/perf_{str_perf_file_name}.csv', index=False)
    logger.info("Strategies' performance saved!")

