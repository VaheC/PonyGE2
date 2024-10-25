from fitness.base_ff_classes.base_ff import base_ff
# import time
import pandas as pd
from pathlib import Path
# import numpy as np
# import re
# import os

def generate_data():
    # df = pd.read_csv(Path(r'C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\Upwork/\AlgoT_ML_Dev/\GrammarEvolution/\PonyGE2/\datasets/\BTCUSD_ohlcv.csv'))
    # df = pd.read_csv(Path(r'C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\Upwork/\AlgoT_ML_Dev/\GrammarEvolution/\PonyGE2/\datasets/\BTC-ETH-1m.csv'))
    df = pd.read_csv(Path(r'C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\Upwork/\AlgoT_ML_Dev/\GrammarEvolution/\PonyGE2/\datasets/\all_data_1min.csv'))
    # df = pd.read_csv('/kaggle/input/btcusd-test/BTCUSD_ohlcv.csv')
    # df = pd.read_csv('/kaggle/input/btcusd-test/BTC-ETH-1m.csv')
    # df = pd.read_csv('/kaggle/input/btcusd-test/all_data_1min.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    # df = df.iloc[-10080:]
    # df = df.iloc[-525600:]
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

        try:
            # t0 = time.time()
            exec(p, d)
            # t1 = time.time()
            fitness = d['fitness']
        except:
            fitness = 404
            
        return fitness