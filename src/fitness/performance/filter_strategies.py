import pandas as pd
from pathlib import Path
from collections import defaultdict
import gc
import os
from tqdm import tqdm
from .testing_func import (calculate_mean_win_perc_entry_testing, calculate_mean_win_perc_exit_testing,
                          calculate_mean_win_perc_core_testing, calculate_mean_performance, 
                          calculate_mc_performance, get_entry_win_pc_df, get_exit_win_pc_df,
                          get_core_win_pc_df, get_perf_df, get_mc_df)

def generate_fold_data(data_path, fold=1, n_bars=50400):
    
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.iloc[n_bars*fold:(fold+1)*n_bars]
    df.sort_values('datetime', ascending=True, inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df

def generate_data(data_path):
    
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.sort_values('datetime', ascending=True, inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df

def generate_strategy_data(strategy_file_path):

    try:
        df_str = pd.read_csv(strategy_file_path)
    except:
        df_str = pd.read_csv(strategy_file_path, sep=';')
        
    df_str = df_str[df_str['fitness'] < 0]
    df_str = df_str[~df_str.duplicated()]

    df_str_get1000 = df_str[df_str['fitness'] <= -1000]#.reset_index(drop=True).iloc[:1000]
    df_str_lt1000 = df_str[df_str['fitness'] > -1000].reset_index(drop=True).iloc[:1000]
    # df_str_small_fitness = df_str[(df_str['fitness'] > -100) & (df_str['fitness'] < -50)]

    df_str = pd.concat([df_str_get1000, df_str_lt1000], axis=0)
    # df_str = pd.concat([df_str, df_str_small_fitness], axis=0)

    df_str.sort_values('fitness', ascending=True, inplace=True)
    df_str.reset_index(drop=True, inplace=True)
    df_str['strategy'] = [f'strategy{i+1}' for i in range(df_str.shape[0])]

    return df_str

def create_txt_code1(buy_signal_txt, buy_exit_txt, sell_signal_txt, sell_exit_txt, 
                     fee=0.015, slippage=0.00005, inv_amount=700000, trade_size=0.5, max_lag=99):

    text_code = f'''import os
CUR_DIR = os.getcwd()
# os.chdir('src')
#import pandas as pd
import numpy as np
import gc
from fitness.indicators import numba_indicators, signals
from fitness.performance.helper_func import merge_buy_sell_pnl, get_drawdowns, get_pnl, get_lag
from fitness.performance.helper_func import trading_signals_buy, trading_signals_sell, change_exit
# os.chdir(CUR_DIR)
#from numba import njit
COMMISSION = {fee}
SLIPPAGE = {slippage}
AVAILABLE_CAPITAL = {inv_amount}
TRADE_SIZE = {trade_size}
MAX_LAG = {max_lag}
buy_idxs, buy_exit_idxs = trading_signals_buy(buy_signal={buy_signal_txt}, exit_signal={buy_exit_txt})
sell_idxs, sell_exit_idxs = trading_signals_sell(sell_signal={sell_signal_txt}, exit_signal={sell_exit_txt})
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

def get_important_stats(df_str, df_data, bars_per_span = 7 * 60 * 24 * 5, n_bars_per_data = 7 * 60 * 24 * 5, 
                        create_txt_code=create_txt_code1):

    final_entry_win_pc_df = pd.DataFrame()
    final_exit_win_pc_df = pd.DataFrame()
    final_core_win_pc_df = pd.DataFrame()
    final_perf_df = pd.DataFrame()
    final_mc_df = pd.DataFrame()

    equity_curve_dict = {}

    if df_data.shape[0] < n_bars_per_data:
        n_bars_per_data = df_data.shape[0]

    strategy_idx = 1

    equity_curve_dict = defaultdict(list)

    for row in tqdm(df_str.itertuples()):

        buy_signal_txt = row.buy
        buy_exit_txt = row.exit_buy
        sell_signal_txt = row.sell
        sell_exit_txt = row.exit_sell

        text_code = create_txt_code(buy_signal_txt, buy_exit_txt, sell_signal_txt, sell_exit_txt)
        
        entry_test_n_not_worked = 0
        exit_test_n_not_worked = 0
        core_test_n_not_worked = 0
        perf_n_not_worked = 0
        mc_n_not_worked = 0
        n_total_cases = 0

        entry_walk_forward_dict = defaultdict(list)

        exit_walk_forward_dict = defaultdict(list)

        core_walk_forward_dict = defaultdict(list)

        performance_walk_forward_dict = defaultdict(list)

        mc_walk_forward_dict = defaultdict(list)

        for idx in range(0, n_bars_per_data, bars_per_span):

            n_total_cases += 1

            # df = df_data.iloc[idx:idx+bars_per_span, :]
            # df.reset_index(drop=True, inplace=True)
            df = df_data.iloc[idx:idx+bars_per_span, :]
            df.reset_index(drop=True, inplace=True)

            price_data = {}
            for col in df.columns:
                if col == 'datetime':
                    continue
                else:
                    price_data[col] = df[col].values
            price_data['day_of_week'] = (df['datetime'].dt.dayofweek + 1).values
            price_data['month'] = df['datetime'].dt.month.values
            price_data['hour'] = df['datetime'].dt.hour.values
            price_data['minute'] = df['datetime'].dt.minute.values

            exec_dict = {'price_data': price_data}
            try:
                exec(text_code, exec_dict)
            except:
                continue

            try:
                equity_curve_arr = exec_dict['equity_curve_arr']
                equity_curve_dict[strategy_idx].append(equity_curve_arr)
            except:
                pass

            try:
                fixed_winning_percent, fixed_bar_winning_percent, random_winning_percent = calculate_mean_win_perc_entry_testing(exec_dict, df)
                entry_walk_forward_dict['fixed_sp_testing'].append(fixed_winning_percent)
                entry_walk_forward_dict['fixed_bar_testing'].append(fixed_bar_winning_percent)
                entry_walk_forward_dict['random_exit_testing'].append(random_winning_percent)
            except:
                entry_test_n_not_worked += 1

            try:
                trend_winning_percent, countertrend_winning_percent, random_winning_percent = calculate_mean_win_perc_exit_testing(exec_dict, df)
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

        temp_signal_df = pd.DataFrame({'strategy': f'strategy{strategy_idx}', 'buy': [buy_signal_txt], 'sell': [sell_signal_txt]})

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

        gc.collect()

        strategy_idx += 1

    final_entry_win_pc_df.reset_index(drop=True, inplace=True)
    final_exit_win_pc_df.reset_index(drop=True, inplace=True)
    final_core_win_pc_df.reset_index(drop=True, inplace=True) 
    final_perf_df.reset_index(drop=True, inplace=True)
    final_mc_df.reset_index(drop=True, inplace=True)

    return final_entry_win_pc_df, final_exit_win_pc_df, final_core_win_pc_df, final_perf_df, final_mc_df

def save_stats(data_path, strategy_file_path, n_fold, logger, stats_path='testing_results', stats_file_name='baseline', n_bars=50400):

    logger.info('Starting data loading process...')
    if n_fold == 0:
        df_data = generate_data(data_path)
        n_bars_per_data = df_data.shape[0]
    else:
        df_data = generate_fold_data(data_path, fold=n_fold, n_bars=n_bars)
        n_bars_per_data = n_bars
    logger.info('Data loading process completed!')

    logger.info("Starting strategies' loading process...")
    df_str = generate_strategy_data(strategy_file_path)
    logger.info('Strategies are loaded!')

    # getting the stats
    logger.info('Starting testing stats calculation...')
    (final_entry_win_pc_df, final_exit_win_pc_df, 
     final_core_win_pc_df, final_perf_df, final_mc_df) = get_important_stats(df_str, df_data, bars_per_span = n_bars, n_bars_per_data=n_bars_per_data)
    logger.info('Testing stats calculation completed!')

    logger.info("Creating %s directory if it doesn't exist...", stats_path)
    if not os.path.exists(stats_path):
        os.mkdir(stats_path)
    logger.info('%s directory creation completed!', stats_path)

    logger.info('Saving stats to %s directory...', stats_path)
    final_entry_win_pc_df.to_csv(f'{stats_path}/entry_testing_{stats_file_name}.csv', index=False)
    final_exit_win_pc_df.to_csv(f'{stats_path}/exit_testing_{stats_file_name}.csv', index=False)
    final_core_win_pc_df.to_csv(f'{stats_path}/core_testing_{stats_file_name}.csv', index=False)
    final_perf_df.to_csv(f'{stats_path}/perf_{stats_file_name}.csv', index=False)
    final_mc_df.to_csv(f'{stats_path}/mc_{stats_file_name}.csv', index=False)
    logger.info('Stats saved!')

def filter_save_strategies(strategy_file_path, logger, stats_path='testing_results', 
                           stats_file_name='baseline', str_path='selected_strategies', 
                           str_file_name='baseline', entry_testing_threshold=50,
                           exit_testing_threshold=50, core_testing_threshold=60,
                           prob_threshold=0.98):

    logger.info('Loading strategies...')
    df_str = generate_strategy_data(strategy_file_path)
    logger.info('Strategies loaded!')

    logger.info('Loading saved testing stats...')
    final_entry_win_pc_df = pd.read_csv(f'{stats_path}/entry_testing_{stats_file_name}.csv')
    final_exit_win_pc_df = pd.read_csv(f'{stats_path}/exit_testing_{stats_file_name}.csv')
    final_core_win_pc_df = pd.read_csv(f'{stats_path}/core_testing_{stats_file_name}.csv')
    final_perf_df = pd.read_csv(f'{stats_path}/perf_{stats_file_name}.csv')
    final_mc_df = pd.read_csv(f'{stats_path}/mc_{stats_file_name}.csv')
    logger.info('Saved testing stats loaded!')
    
    # getting strategies that have winning percentage equal to 50 or more for entry testing cases
    logger.info('Filtering strategies based on entry testing...')
    entry_testing_strategies = final_entry_win_pc_df[
        (final_entry_win_pc_df['Fixed_StopLoss_TakeProfit_testing'] >= entry_testing_threshold) & 
        (final_entry_win_pc_df['Fixed_Bar_testing'] >= entry_testing_threshold) & 
        (final_entry_win_pc_df['Random_Exit_testing'] >= entry_testing_threshold)
    ]['strategy'].tolist()
    final_entry_win_pc_df[final_entry_win_pc_df['strategy'].isin(entry_testing_strategies)]
    logger.info('Entry testing filtering completed!')

    # getting strategies that have passed entry testing and have winning percentage 
    # equal to 50 or more (except for Random_Entry_testing)
    logger.info('Filtering survived strategies using exit testing...')
    exit_testing_strategies = final_exit_win_pc_df[
        (final_exit_win_pc_df['Trend_testing'] >= exit_testing_threshold) & 
        (final_exit_win_pc_df['Countertrend_testing'] >= exit_testing_threshold) & 
        # (final_exit_win_pc_df['Random_Entry_testing'] >= exit_testing_threshold) &
        (final_exit_win_pc_df['strategy'].isin(entry_testing_strategies))
    ]['strategy'].tolist()
    logger.info('Exit testing filtering completed!')

    # getting strategies that has passed both entry and exit testing and have winning percentage equal to 60 or more
    logger.info('Filtering survived strategies using core testing...')
    core_testing_strategies = final_core_win_pc_df[
        (final_core_win_pc_df['Core_Testing'] >= core_testing_threshold) &
        (final_core_win_pc_df['strategy'].isin(exit_testing_strategies))
    ]['strategy'].tolist()
    logger.info('Core testing filtering completed!')

    # getting performance for strategies that has passed all tests and have positive ROI
    logger.info('Filtering survived strategies using ROI...')
    perf_strategies = final_perf_df[
        (final_perf_df['strategy'].isin(core_testing_strategies)) & 
        (final_perf_df['ROI (%)'] > 0)
    ]['strategy'].tolist()
    logger.info('ROI filtering completed!')

    # getting simulation results for strategies that has passed all tests and have positive ROI
    logger.info('Filtering survived strategies using Monte Carlo...')
    mc_strategies = final_mc_df[
        (final_mc_df['strategy'].isin(perf_strategies)) &
        (final_mc_df['prob'] > prob_threshold) &
        (final_mc_df['ratio'] > 1)
    ]['strategy'].values
    logger.info('Monte Carlo filtering completed!')

    # Saving the selected strategies
    if len(mc_strategies) == 0:

        logger.info('There are not survived strategies!')

    else:

        logger.info("Creating %s directory if it doesn't exist...", str_path)
        if not os.path.exists(str_path):
            os.mkdir(str_path)
        logger.info("%s directory created!", str_path)

        logger.info('Saving survived strategies to %s directory...', str_path)
        df_selected_str = df_str[df_str['strategy'].isin(mc_strategies)]
        selected_path = f'{str_path}/selected_strategies_{str_file_name}.csv'
        df_selected_str.to_csv(selected_path, index=False)
        logger.info('Saving survived strategies to %s directory completed!', str_path)
