import numpy as np
from numba import njit
import pandas as pd
import polars as pl
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import os
import gc

import warnings
warnings.filterwarnings('ignore')

# Limited testing

@njit(cache=True)
def get_signals(signal_list, exit_list):

    start_idx = 0
    exit_idx = 0
   
    for i in range(len(signal_list[0])):

        if i == start_idx:

            if signal_list[0][i] == 0:

                start_idx += 1

                exit_list[0][i] = 0

            else:

                for j in range(i+1, len(exit_list[0])):
                    if exit_list[0][j] == -signal_list[0][i]:
                        exit_idx = j
                        break
                    else:
                        exit_idx = j

                for k in range(i+1, exit_idx+1):
                    if signal_list[0][k] == -signal_list[0][i]:
                        exit_idx = k
                        exit_list[1][k] = signal_list[1][k]
                        break
                    else:
                        exit_idx = k

                for p in range(i+1, exit_idx):
                    signal_list[0][p] = 0
                    exit_list[0][p] = 0
                
                if exit_idx == len(signal_list[0]) - 1 and exit_list[0][exit_idx] != -signal_list[0][i]:
                    exit_list[0][exit_idx] = 0
                    exit_list[0][i] = 0
                    signal_list[0][exit_idx] = 0
                else:
                    exit_list[0][exit_idx] = -signal_list[0][i]
                    exit_list[0][i] = 0
                    signal_list[0][exit_idx] = 0
                
                start_idx = exit_idx + 1

        else:

            continue

    if sum(np.abs(signal_list[0])) == sum(np.abs(exit_list[0])):

        return signal_list, exit_list
    
    else:

        for i in range(len(signal_list[0])):
            if signal_list[0][-(i+1)] != 0:
                signal_list[0][-(i+1)] = 0
                break

        return signal_list, exit_list
    
@njit(cache=True)
def create_position_open_prices(signal_list, exit_list):

    pos_open_prices = np.zeros(len(signal_list[0]))
    pos_exit_prices = np.zeros(len(exit_list[0]))

    start_idx = 0
    price_idx = 0

    for i in range(len(signal_list[0])):
        if exit_list[0][i] != 0:
            for j in range(start_idx, i):
                if signal_list[0][j] != 0:
                    price_idx = j
                    break
            pos_open_prices[i] = signal_list[1][price_idx]
            pos_exit_prices[i] = exit_list[1][i]
            start_idx = i
        else:
            pass

    return pos_open_prices, pos_exit_prices

@njit(cache=True)
def get_pnl_testing(
    trade_close_prices,
    signal_list, 
    trade_open_prices,
    commission=0.015,
    slippage=0.05,
    init_inv=20000,
    trade_size=0.1
):

    pnl_list = np.zeros(len(trade_close_prices))

    for i in range(len(trade_close_prices)):

        if signal_list[i] == 0 or trade_open_prices[i] == 0:
            pass
        
        # signal_list contains the points where exit occurs
        elif signal_list[i] == -1: 
            temp_n_assets = int(init_inv * trade_size / trade_open_prices[i])
            temp_pnl = temp_n_assets * (trade_close_prices[i] - trade_open_prices[i] * (1 + slippage)) 
            temp_pnl = temp_pnl * (1 - commission)
            init_inv += temp_pnl

        else:
            temp_n_assets = int(init_inv * trade_size / trade_open_prices[i])
            temp_pnl = temp_n_assets * (trade_open_prices[i] * (1 - slippage) - trade_close_prices[i])
            temp_pnl = temp_pnl * (1 - commission)
            init_inv += temp_pnl

        pnl_list[i] = temp_pnl

    return pnl_list

# Entry testing

# Fixed stop and target exit

@njit(cache=True)
def get_exit_entry_testing1(
    close_prices, 
    open_prices,  
    signal_list,
    stoploss_th,
    takeprofit_th, 
    commission, 
    slippage, 
    init_inv, 
    trade_size
):

    exit_list = np.zeros((2, len(close_prices)))

    for i in range(len(close_prices)-1):

        if signal_list[1][i] == 0:

            pass

        else:

            temp_n_assets = int(init_inv * trade_size / signal_list[1][i])
            if signal_list[0][i] == 1:
                temp_pnl = temp_n_assets * (close_prices[i] - signal_list[1][i] * (1 + slippage))
            else:
                temp_pnl = -temp_n_assets * (close_prices[i] - signal_list[1][i] * (1 - slippage))
            temp_pnl = temp_pnl * (1 - commission)
            init_inv += temp_pnl

            if -temp_pnl >= stoploss_th or temp_pnl >= takeprofit_th:
                exit_list[0][i+1] = -signal_list[0][i]
                exit_list[1][i+1] = open_prices[i+1]
            else:
                pass
        
    return exit_list

# Fixed bar exit

@njit(cache=True)
def get_exit_entry_testing2( 
    open_prices,  
    signal_list,
    n_exit_bars
):

    exit_list = np.zeros((2, len(signal_list[0])))

    n_exit_bars = np.int64(n_exit_bars)

    for i in range(len(signal_list[0])-1):

        if signal_list[0][i] == 0:

            pass

        else:
            
            if i + n_exit_bars < len(signal_list[0]):
                exit_list[0][i+n_exit_bars] = -signal_list[0][i]
                exit_list[1][i+n_exit_bars] = open_prices[i+n_exit_bars]
            else:
                pass
        
    return exit_list

# Random exit

@njit(cache=True)
def get_exit_entry_testing3( 
    open_prices,  
    signal_list
):

    exit_list = np.zeros((2, len(signal_list[0])))

    for i in range(len(signal_list[0])-1):

        if signal_list[0][i] == 0:

            pass

        else:

            for j in range(i+1, len(signal_list[0])):
                if signal_list[0][j] != 0:
                    j = j - 1
                    break
                else:
                    if np.random.rand() > 0.5:
                        break
            
            exit_list[0][j] = -signal_list[0][i]
            exit_list[1][j] = open_prices[j]
        
    return exit_list

# Winning percentage

def calculate_mean_win_perc_entry_testing(exec_dict, df):

    commission = exec_dict['COMMISSION']
    slippage = exec_dict['SLIPPAGE'] 
    init_inv = exec_dict['AVAILABLE_CAPITAL']
    trade_size = exec_dict['TRADE_SIZE'] 

    if len(exec_dict['buy_idxs']) != 0:
        signal_idxs = list(exec_dict['buy_idxs'])
        if len(exec_dict['sell_idxs']) != 0:
            signal_idxs.extend(list(exec_dict['sell_idxs']))
    else:
        if len(exec_dict['sell_idxs']) != 0:
            signal_idxs = list(exec_dict['sell_idxs'])
        else:
            signal_idxs = []

    if len(signal_idxs) != 0:
        signal_idxs = sorted(signal_idxs)
        signal_idxs_true = [i - 1 for i in signal_idxs]
    else:
        signal_idxs_true = []

    df['buy'] = 0
    if len(exec_dict['buy_idxs']) != 0:
        buy_idxs = [i - 1 for i in list(exec_dict['buy_idxs'])]
        df.loc[df.index.isin(buy_idxs), 'buy'] = 1

    df['sell'] = 0
    if len(exec_dict['sell_idxs']) != 0:
        sell_idxs = [i - 1 for i in list(exec_dict['sell_idxs'])]
        df.loc[df.index.isin(sell_idxs), 'sell'] = -1

    df['signal'] = df['buy'] + df['sell']

    df['new_signal'] = 0
    df['signal_prices'] = 0
    if len(signal_idxs) != 0:
        df.loc[df.index.isin(signal_idxs_true), 'new_signal'] = df.loc[df.index.isin(signal_idxs_true), 'signal'].values
        df.loc[df.index.isin(signal_idxs), 'signal_prices'] = df.loc[df.index.isin(signal_idxs), 'btc_open'].values

    signal_list = np.zeros((2, df.shape[0]))
    signal_list[0][1:] = df['new_signal'].values[:-1]
    signal_list[1] = df['signal_prices'].values

    # fixed stop and target exit testing
    exit_list = get_exit_entry_testing1(
        close_prices=df['btc_close'].values, 
        open_prices=df['btc_open'].values,  
        signal_list=signal_list,
        stoploss_th=50,
        takeprofit_th=100,
        commission=commission, 
        slippage=slippage, 
        init_inv=init_inv, 
        trade_size=trade_size
    )

    signal_list, exit_list = get_signals(signal_list, exit_list)
    pos_open_prices, pos_exit_prices = create_position_open_prices(signal_list, exit_list)

    pnl_list = get_pnl_testing(
        trade_close_prices=pos_exit_prices,
        signal_list=exit_list[0], 
        trade_open_prices=pos_open_prices,
        commission=commission, 
        slippage=slippage, 
        init_inv=init_inv, 
        trade_size=trade_size
    )

    fixed_winning_percent = 100 * sum(pnl_list > 0) / np.sum(pnl_list != 0)

    # fixed bar exit testing
    exit_list = get_exit_entry_testing2( 
        open_prices=df['btc_open'].values,  
        signal_list=signal_list,
        n_exit_bars=5
    )

    signal_list, exit_list = get_signals(signal_list, exit_list)
    pos_open_prices, pos_exit_prices = create_position_open_prices(signal_list, exit_list)

    pnl_list = get_pnl_testing(
        trade_close_prices=pos_exit_prices,
        signal_list=exit_list[0], 
        trade_open_prices=pos_open_prices,
        commission=commission, 
        slippage=slippage, 
        init_inv=init_inv, 
        trade_size=trade_size
    )

    fixed_bar_winning_percent = 100 * sum(pnl_list > 0) / np.sum(pnl_list != 0)

    # random exit testing
    exit_list = get_exit_entry_testing3( 
        open_prices=df['btc_open'].values,  
        signal_list=signal_list
    )

    signal_list, exit_list = get_signals(signal_list, exit_list)
    pos_open_prices, pos_exit_prices = create_position_open_prices(signal_list, exit_list)

    pnl_list = get_pnl_testing(
        trade_close_prices=pos_exit_prices,
        signal_list=exit_list[0], 
        trade_open_prices=pos_open_prices,
        commission=commission, 
        slippage=slippage, 
        init_inv=init_inv, 
        trade_size=trade_size
    )

    random_winning_percent = 100 * sum(pnl_list > 0) / np.sum(pnl_list != 0)

    return fixed_winning_percent, fixed_bar_winning_percent, random_winning_percent

def get_entry_win_pc_df(entry_walk_forward_dict, n_not_worked, n_total_cases):
    
    entry_mean_win_perc_dict = defaultdict(list)

    entry_mean_win_perc_dict['Fixed_StopLoss_TakeProfit_testing'].\
        append(np.nanmean(entry_walk_forward_dict['fixed_sp_testing']))
    entry_mean_win_perc_dict['Fixed_Bar_testing'].\
        append(np.nanmean(entry_walk_forward_dict['fixed_bar_testing']))
    entry_mean_win_perc_dict['Random_Exit_testing'].\
        append(np.nanmean(entry_walk_forward_dict['random_exit_testing']))
    entry_mean_win_perc_dict['Not_Working'].\
        append(100 * n_not_worked / n_total_cases)
        
    entry_win_pc_df = pd.DataFrame(entry_mean_win_perc_dict)

    return entry_win_pc_df

# Exit testing

# Similar approach entry

@njit(cache=True)
def get_entry_exit_testing1(
    close_prices,
    open_prices,
    n_bars
):

    signal_list = np.zeros((2, len(close_prices)))

    # n_bars = np.int64(n_bars)

    for i in range(n_bars, len(signal_list[0])):
            
        if close_prices[i - n_bars] - close_prices[i - 1] > 0:
            signal_list[0][i] = 1
            signal_list[1][i] = open_prices[i]
        elif close_prices[i - n_bars] - close_prices[i - 1] < 0:
           signal_list[0][i] = -1
           signal_list[1][i] = open_prices[i]
        else:
            pass
        
    return signal_list

@njit(cache=True)
def get_rsi(close_prices, prev_close_prices, length=14):
    # Create numpy arrays to store the gain/loss values
    gains = np.zeros(len(close_prices))
    losses = np.zeros(len(close_prices))

    # Iterate through the data frame and calculate the gain/loss for each period
    for i in range(1, len(close_prices)):
        change = close_prices[i] - prev_close_prices[i]
        if change > 0:
            gains[i] = change
        elif change < 0:
            losses[i] = abs(change)

    # Calculate the average gain and loss for each period
    avg_gains = np.zeros(len(close_prices))
    avg_losses = np.zeros(len(close_prices))
    for i in range(length, len(close_prices)):
        avg_gains[i] = np.mean(gains[i-length:i])
        avg_losses[i] = np.mean(losses[i-length:i])

    # Calculate the relative strength and RSI for each period
    rs = np.zeros(len(close_prices))
    rsi = np.zeros(len(close_prices))
    
    for i in range(len(close_prices)):
        if i+1 < length:
            rsi[i] = np.nan
        elif avg_losses[i] == 0:
            rs[i] = avg_gains[i]
            rsi[i] = 100
        else:
            rs[i] = avg_gains[i] / avg_losses[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))

    return rsi

@njit(cache=True)
def get_entry_exit_testing2(
    close_prices, open_prices, prev_close_prices, rsi_window_size, rsi_threshold
):

    signal_list = np.zeros((2, len(close_prices)))

    rsi = get_rsi(close_prices, prev_close_prices, length=rsi_window_size)

    for i in range(len(close_prices)-1):

        if i < rsi_window_size - 1 or np.isnan(rsi[i]):
            continue
       
        if rsi[i] < rsi_threshold:
            signal_list[0][i+1] = 1
            signal_list[1][i+1] = open_prices[i+1]
        elif rsi[i] > (100 - rsi_threshold):
            signal_list[0][i+1] = -1
            signal_list[1][i+1] = open_prices[i+1]
        else:
            pass

    return signal_list

# Random entry

@njit(cache=True)
def get_entry_exit_testing3(
    open_prices
):

    signal_list = np.zeros((2, len(open_prices)))

    for i in range(len(open_prices)-1):

        if np.random.rand() > 0.7:
            signal_list[0][i] = 1
            signal_list[1][i] = open_prices[i]
        elif np.random.rand() < 0.3:
            signal_list[0][i] = -1
            signal_list[1][i] = open_prices[i]
        else:
            pass

    return signal_list

# Winning percentage

def calculate_mean_win_perc_exit_testing(exec_dict, df):

    commission = exec_dict['COMMISSION']
    slippage = exec_dict['SLIPPAGE'] 
    init_inv = exec_dict['AVAILABLE_CAPITAL']
    trade_size = exec_dict['TRADE_SIZE']       

    if len(exec_dict['sell_exit_idxs']) != 0:
        signal_idxs = list(exec_dict['sell_exit_idxs'])
        if len(exec_dict['buy_exit_idxs']) != 0:
            signal_idxs.extend(list(exec_dict['buy_exit_idxs']))
    else:
        if len(exec_dict['buy_exit_idxs']) != 0:
            signal_idxs = list(exec_dict['buy_exit_idxs'])
        else:
            signal_idxs = []

    if len(signal_idxs) != 0:
        signal_idxs = sorted(signal_idxs)
        signal_idxs_true = [i - 1 for i in signal_idxs]
    else:
        signal_idxs_true = []

    df['buy'] = 0
    if len(exec_dict['sell_exit_idxs']) != 0:
        buy_idxs = [i - 1 for i in list(exec_dict['sell_exit_idxs'])]
        df.loc[df.index.isin(buy_idxs), 'buy'] = 1

    df['sell'] = 0
    if len(exec_dict['buy_exit_idxs']) != 0:
        sell_idxs = [i - 1 for i in list(exec_dict['buy_exit_idxs'])]
        df.loc[df.index.isin(sell_idxs), 'sell'] = -1

    df['signal'] = df['buy'] + df['sell']

    df['exit_signal'] = 0
    df['exit_prices'] = 0
    if len(signal_idxs) != 0:
        df.loc[df.index.isin(signal_idxs_true[1:]), 'exit_signal'] = df.loc[df.index.isin(signal_idxs_true[1:]), 'signal'].values
        df.loc[df.index.isin(signal_idxs[1:]), 'exit_prices'] = df.loc[df.index.isin(signal_idxs[1:]), 'btc_open'].values

    exit_list = np.zeros((2, df.shape[0]))
    exit_list[0][1:] = df['exit_signal'].values[:-1]
    exit_list[1] = df['exit_prices'].values

    # replacing entry with trend following entry
    signal_list = get_entry_exit_testing1(
        close_prices=df['btc_close'].values,
        open_prices=df['btc_open'].values,
        n_bars=5
    )

    signal_list, exit_list = get_signals(signal_list, exit_list)
    pos_open_prices, pos_exit_prices = create_position_open_prices(signal_list, exit_list)

    pnl_list = get_pnl_testing(
        trade_close_prices=pos_exit_prices,
        signal_list=exit_list[0], 
        trade_open_prices=pos_open_prices,
        commission=commission, 
        slippage=slippage, 
        init_inv=init_inv, 
        trade_size=trade_size
    )

    trend_winning_percent = 100 * sum(pnl_list > 0) / np.sum(pnl_list != 0)

    # replacing entry with countertrend entry
    signal_list = get_entry_exit_testing2(
        close_prices=df['btc_close'].values, 
        open_prices=df['btc_open'].values, 
        prev_close_prices=df['btc_close'].shift(1).fillna(method='bfill').values, 
        rsi_window_size=10, 
        rsi_threshold = 20
    )

    signal_list, exit_list = get_signals(signal_list, exit_list)
    pos_open_prices, pos_exit_prices = create_position_open_prices(signal_list, exit_list)

    pnl_list = get_pnl_testing(
        trade_close_prices=pos_exit_prices,
        signal_list=exit_list[0], 
        trade_open_prices=pos_open_prices,
        commission=commission, 
        slippage=slippage, 
        init_inv=init_inv, 
        trade_size=trade_size
    )

    countertrend_winning_percent = 100 * sum(pnl_list > 0) / np.sum(pnl_list != 0)

    # random entry testing
    signal_list = get_entry_exit_testing3(
        open_prices=df['btc_open'].values
    )

    signal_list, exit_list = get_signals(signal_list, exit_list)
    pos_open_prices, pos_exit_prices = create_position_open_prices(signal_list, exit_list)

    pnl_list = get_pnl_testing(
        trade_close_prices=pos_exit_prices,
        signal_list=exit_list[0], 
        trade_open_prices=pos_open_prices,
        commission=commission, 
        slippage=slippage, 
        init_inv=init_inv, 
        trade_size=trade_size
    )

    random_winning_percent = 100 * sum(pnl_list > 0) / np.sum(pnl_list != 0)


    return trend_winning_percent, countertrend_winning_percent, random_winning_percent

def get_exit_win_pc_df(exit_walk_forward_dict, n_not_worked, n_total_cases):

    exit_mean_win_perc_dict = defaultdict(list)

    exit_mean_win_perc_dict['Trend_testing'].\
        append(np.nanmean(exit_walk_forward_dict['trend_entry_testing']))
    exit_mean_win_perc_dict['Countertrend_testing'].\
        append(np.nanmean(exit_walk_forward_dict['countertrend_entry_testing']))
    exit_mean_win_perc_dict['Random_Entry_testing'].\
        append(np.nanmean(exit_walk_forward_dict['random_entry_testing']))
    exit_mean_win_perc_dict['Not_Working'].\
        append(100 * n_not_worked / n_total_cases)

    exit_win_pc_df = pd.DataFrame(exit_mean_win_perc_dict)

    return exit_win_pc_df

# Core testing

# Winning percentage

def calculate_mean_win_perc_core_testing(exec_dict):
            
    pnl_list = exec_dict['all_arr']

    winning_percent = 100 * sum(pnl_list > 0) / np.sum(pnl_list != 0)
    
    return winning_percent

def get_core_win_pc_df(core_walk_forward_dict, n_not_worked, n_total_cases):

    core_mean_win_perc_dict = defaultdict(list)
    core_mean_win_perc_dict['Core_Testing'].\
        append(np.nanmean(core_walk_forward_dict['core_testing']))
    core_mean_win_perc_dict['Not_Working'].\
        append(100 * n_not_worked / n_total_cases)

    core_win_pc_df = pd.DataFrame(core_mean_win_perc_dict)

    return core_win_pc_df

# Performance

@njit(cache=True)
def get_drawdown(pnl_list):
    max_dd = 0.0
    peak = pnl_list[0]

    for i in range(1, len(pnl_list)):
        if pnl_list[i] > peak:
            peak = pnl_list[i]
        drawdown = 100 * (peak - pnl_list[i]) / peak
        if drawdown > max_dd:
            max_dd = drawdown

    return max_dd

@njit(cache=True)
def get_drawdown_duration(pnl_arr):
    peak = pnl_arr[0]
    max_duration = 0
    current_duration = 0

    for pnl in pnl_arr:
        if pnl < peak:  # We are in a drawdown
            current_duration += 1
        else:  # We found a new peak
            peak = pnl
            current_duration = 0

        max_duration = max(max_duration, current_duration)

    return max_duration

@njit(cache=True)
def get_sharpe_ratio(
    pnl_list, 
    risk_free_rate=0
):
    # mean_return = np.mean(pnl_list)
    mean_return = 0
    for i in range(len(pnl_list)):
        mean_return += pnl_list[i]

    if len(pnl_list) == 0:
        mean_return = 0
    else:
        mean_return = mean_return / len(pnl_list)

    # std_dev = np.std(pnl_list)
    std_dev = 0
    for i in range(len(pnl_list)):
        std_dev += (pnl_list[i] - mean_return) ** 2

    if len(pnl_list) == 1:
        std_dev = np.sqrt(std_dev/len(pnl_list))
    else:
        std_dev = np.sqrt(std_dev/(len(pnl_list) - 1))

    if std_dev == 0:
        std_dev += 0.0001

    sharpe_ratio = (mean_return - risk_free_rate) / std_dev
    return sharpe_ratio

@njit(cache=True)
def get_sortino_ratio(profit_loss, target_pl=0):

    downside_pnl = profit_loss[profit_loss < target_pl]
    if len(downside_pnl) == 0:
        return 0

    downside_deviation = np.sqrt(np.mean((downside_pnl - target_pl) ** 2))
    if downside_deviation == 0:
        return 0

    mean_pnl = np.mean(profit_loss)
    sortino_ratio = (mean_pnl - target_pl) / downside_deviation

    return sortino_ratio

def calculate_mean_performance(exec_dict, monkey_test=False):

    pnl_list = exec_dict['all_arr']

    init_inv = exec_dict['AVAILABLE_CAPITAL']
    trade_size = exec_dict['TRADE_SIZE']       

    signal_idxs = list(exec_dict['buy_idxs'])
    signal_idxs.extend(list(exec_dict['sell_idxs']))
    signal_idxs = sorted(signal_idxs)

    metric_dict = {}

    metric_dict['n_trades'] = len(signal_idxs) - 1
    metric_dict['overall_pnl'] = np.sum(pnl_list)
    metric_dict['roi'] = 100 * metric_dict['overall_pnl'] / (trade_size * init_inv)
    metric_dict['avg_drawdown'] = exec_dict['avg_drawdown']
    metric_dict['max_dd'] = get_drawdown(pnl_list)
    metric_dict['drawdown_dur'] = get_drawdown_duration(pnl_list)
    metric_dict['pnl_avgd_ratio'] = exec_dict['fitness']

    annualized_sharpe_ration = np.sqrt(525600) * get_sharpe_ratio(pnl_list, risk_free_rate=0)
    metric_dict['sharpe_ratio'] = annualized_sharpe_ration

    annualized_sortino_ratio = np.sqrt(525600) * get_sortino_ratio(pnl_list)
    metric_dict['sortino_ratio'] = annualized_sortino_ratio

    if monkey_test:

        pnl_mren_arr = exec_dict['pnl_mren_arr']
        max_dd_mren_arr = exec_dict['max_dd_mren_arr']

        pnl_mren_good_cases = np.sum(np.where(metric_dict['overall_pnl'] > pnl_mren_arr, 1, 0))
        metric_dict['mt_pnl'] = 100 * pnl_mren_good_cases / len(pnl_mren_arr)

        max_dd_mren_good_cases = np.sum(np.where(metric_dict['max_dd'] < max_dd_mren_arr, 1, 0))
        metric_dict['mt_mdd'] = 100 * max_dd_mren_good_cases / len(max_dd_mren_arr)

    return metric_dict

def get_perf_df(performance_walk_forward_dict, n_not_worked, n_total_cases):

    mean_perf_dict = defaultdict(list)
    
    mean_perf_dict['N_Trades'].\
        append(np.nanmean(performance_walk_forward_dict['n_trades']))
    mean_perf_dict['PNL'].\
        append(np.nanmean(performance_walk_forward_dict['pnl']))
    mean_perf_dict['ROI (%)'].\
        append(np.nanmean(performance_walk_forward_dict['roi']))
    mean_perf_dict['AVG_Drawdown'].\
        append(np.nanmean(performance_walk_forward_dict['avg_drawdown']))
    mean_perf_dict['Drawdown (%)'].\
        append(np.nanmean(performance_walk_forward_dict['drawdown']))
    mean_perf_dict['Drawdown_Duration'].\
        append(np.nanmean(performance_walk_forward_dict['drawdown_dur']))
    mean_perf_dict['PNL_AVGD_Ratio'].\
        append(np.nanmean(performance_walk_forward_dict['pnl_avgd_ratio']))
    mean_perf_dict['Sharpe_Ratio'].\
        append(np.nanmean(performance_walk_forward_dict['sharpe_ratio']))
    mean_perf_dict['Sortino_Ratio'].\
        append(np.nanmean(performance_walk_forward_dict['sortino_ratio']))
    mean_perf_dict['Not_Working'].\
        append(100 * n_not_worked / n_total_cases)
    
    if 'mt_pnl' in performance_walk_forward_dict.keys():
        mean_perf_dict['MT_PNL_D'].\
            append(np.nanmean(performance_walk_forward_dict['mt_pnl']))
        mean_perf_dict['MT_MDD_D'].\
            append(np.nanmean(performance_walk_forward_dict['mt_mdd']))

    perf_df = pd.DataFrame(mean_perf_dict)

    return perf_df

# Equity curve Monte Carlo simulation

@njit(cache=True)
def get_random_idxs4mc(arr, num_elements):
    # Get the length of the input array
    n = len(arr)

    if num_elements > n:
        raise ValueError("num_elements must be less than or equal to the length of the array.")

    # Generate random indices with replacement
    indices = np.random.randint(0, n, num_elements)

    # Return sorted indices
    return indices

@njit(cache=True)
def run_simulation(pnl_list, n_runs, init_inv, trade_size):

    max_dd_array = np.zeros(n_runs)
    dd_dur_array = np.zeros(n_runs)
    profit_array = np.zeros(n_runs)
    roi_array = np.zeros(n_runs)
    binary_profit_array = np.zeros(n_runs, dtype=np.int32)
    
    for i in range(n_runs):
        
        # Generate bootstrap sample
        temp_random_idxs = get_random_idxs4mc(arr=pnl_list, num_elements=len(pnl_list))
        temp_pnl_list = np.zeros(len(temp_random_idxs))
        for j in range(len(temp_random_idxs)):
            temp_pnl_list[j] = pnl_list[temp_random_idxs[j]]

        equity_curve_list = np.cumsum(temp_pnl_list)
        
        # Max Drawdown
        max_dd = get_drawdown(equity_curve_list)
        max_dd_array[i] = max_dd
        
        # Drawdown Duration
        dd_dur = get_drawdown_duration(equity_curve_list)
        dd_dur_array[i] = dd_dur
        
        # Profit
        profit = np.sum(temp_pnl_list)
        profit_array[i] = profit
        
        # ROI
        roi = 100 * profit / (init_inv * trade_size)
        roi_array[i] = roi
        
        # Binary Profit
        if profit > 0:
            binary_profit_array[i] = 1

    return max_dd_array, dd_dur_array, profit_array, roi_array, binary_profit_array

def get_mc_results(pnl_list, init_inv, trade_size, n_runs):

    max_dd_array, dd_dur_array, profit_array, roi_array, binary_profit_array = run_simulation(pnl_list, n_runs, init_inv, trade_size)

    mc_dict = {}
    mc_dict['median_max_dd'] = np.median(max_dd_array)
    mc_dict['median_dd_dur'] = np.median(dd_dur_array)
    mc_dict['median_profit'] = np.median(profit_array)
    mc_dict['median_return'] = np.median(roi_array)
    mc_dict['return_dd_ratio'] = np.median(roi_array) / np.median(max_dd_array)
    mc_dict['prob_profit'] = np.sum(binary_profit_array) / len(binary_profit_array)

    return mc_dict

def calculate_mc_performance(exec_dict):
            
    pnl_list = exec_dict['all_arr']

    init_inv = exec_dict['AVAILABLE_CAPITAL']
    trade_size = exec_dict['TRADE_SIZE']       

    mc_dict = get_mc_results(pnl_list, init_inv, trade_size, n_runs=10000)

    return mc_dict

def get_mc_df(mc_walk_forward_dict, n_not_worked, n_total_cases):

    mean_mc_dict = defaultdict(list)

    mean_mc_dict['median_drawdown (%)'].\
        append(np.nanmean(mc_walk_forward_dict['median_max_dd']))
    mean_mc_dict['median_drawdown_duration'].\
        append(np.nanmean(mc_walk_forward_dict['median_dd_dur']))
    mean_mc_dict['median_profit'].\
        append(np.nanmean(mc_walk_forward_dict['median_profit']))
    mean_mc_dict['median_ROI (%)'].\
        append(np.nanmean(mc_walk_forward_dict['median_return']))
    mean_mc_dict['ratio'].\
        append(np.nanmean(mc_walk_forward_dict['return_dd_ratio']))
    mean_mc_dict['prob'].\
        append(np.nanmean(mc_walk_forward_dict['prob_profit']))
    mean_mc_dict['Not_Working'].\
        append(100 * n_not_worked / n_total_cases)

    mc_df = pd.DataFrame(mean_mc_dict)

    return mc_df

