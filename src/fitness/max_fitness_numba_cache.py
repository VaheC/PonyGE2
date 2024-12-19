from fitness.base_ff_classes.base_ff import base_ff
# import time
import pandas as pd
from pathlib import Path
# import numpy as np
# import re
# import os
# from fitness.custom_logger.load_logger import create_terminal_logger
# from fitness.performance.helper_func import get_max_drawdown

def generate_data():
    # df = pd.read_csv(Path(r'C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\Upwork/\AlgoT_ML_Dev/\GrammarEvolution/\PonyGE2/\datasets/\BTCUSD_ohlcv.csv'))
    # df = pd.read_csv(Path(r'C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\Upwork/\AlgoT_ML_Dev/\GrammarEvolution/\PonyGE2/\datasets/\BTC-ETH-1m.csv'))
    # df = pd.read_csv(Path(r'C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\Upwork/\AlgoT_ML_Dev/\GrammarEvolution/\PonyGE2/\datasets/\all_data_1min.csv'))
    # df = pd.read_csv(Path(r'C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\Upwork/\AlgoT_ML_Dev/\GrammarEvolution/\PonyGE2/\all_data_1min.csv'))

    # df = pd.read_csv(Path(r'C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\Upwork/\AlgoT_ML_Dev/\GrammarEvolution/\PonyGE2/\data_folds_15min/\data_fold1.csv'))

    df = pd.read_csv(Path(r'C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\Upwork/\AlgoT_ML_Dev/\GrammarEvolution/\PonyGE2/\data_folds_60min/\data_fold1.csv'))
    
    # df = pd.read_csv('/kaggle/input/btcusd-test/BTCUSD_ohlcv.csv')
    # df = pd.read_csv('/kaggle/input/btcusd-test/BTC-ETH-1m.csv')
    # df = pd.read_csv('/kaggle/input/btcusd-test/all_data_1min.csv')
    # df = pd.read_csv(Path(r'C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\Upwork/\AlgoT_ML_Dev/\GrammarEvolution/\PonyGE2/\data_60min.csv'))
    df['datetime'] = pd.to_datetime(df['datetime'])
    # df = df.iloc[-10080:]
    # df = df.iloc[-525600:]
    # df = df.iloc[252000: (252000+50400)]
    df.sort_values('datetime', ascending=True, inplace=True)
    df.reset_index(inplace=True, drop=True)
    price_data = {}
    # price_data['open'] = df['open'].values
    # price_data['close'] = df['close'].values
    # price_data['high'] = df['high'].values
    # price_data['low'] = df['low'].values
    # price_data['volume'] = df['volume'].values
    for col in df.columns:
        if col == 'datetime':
            continue
        else:
            price_data[col] = df[col].values
    price_data['day_of_week'] = (df['datetime'].dt.dayofweek + 1).values
    price_data['month'] = df['datetime'].dt.month.values
    price_data['hour'] = df['datetime'].dt.hour.values
    price_data['minute'] = df['datetime'].dt.minute.values
    # price_data['month'] = df['datetime'].dt.month.values
    # price_data['day_of_year'] = df['datetime'].dt.dayofyear.values
    return price_data

class max_fitness_numba_cache(base_ff):

    def __init__(self):

        super().__init__()

    def evaluate(self, ind, **kwargs):

        p = ind.phenotype

        self.test_data = generate_data()
        d = {'price_data': self.test_data}

        # logger = create_terminal_logger()

        try:
            # t0 = time.time()
            exec(p, d)
            # t1 = time.time()
            fitness = d['fitness']

            # try:
            #     roi = d['pf'].stats()['Total Return [%]']
            #     mdd = d['pf'].stats()['Max Drawdown [%]']
            # except:
            #     roi = 100 * d['equity_curve_arr'][-1] / (d['AVAILABLE_CAPITAL'] * d['TRADE_SIZE'])
            #     mdd = get_max_drawdown(d['equity_curve_arr'])

            # logger.info(f"ROI: {roi:.4f}, MDD: {mdd:.4f}, Fitness: {fitness:.4f}")
        except:
            fitness = 404
            
        return fitness