from fitness.base_ff_classes.base_ff import base_ff
# import random
import time
import pandas as pd
# from fitness.indicators.indicators import *
from pathlib import Path
import numpy as np
# from numba import njit
import re
import os

def generate_data():
    # df = pd.read_csv(Path(r'C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\Upwork/\AlgoT_ML_Dev/\GrammarEvolution/\PonyGE2/\datasets/\BTCUSD_ohlcv.csv'))
    # df = pd.read_csv(Path(r'C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\Upwork/\AlgoT_ML_Dev/\GrammarEvolution/\PonyGE2/\datasets/\BTC-ETH-1m.csv'))
    # df = pd.read_csv(Path(r'C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\Upwork/\AlgoT_ML_Dev/\GrammarEvolution/\PonyGE2/\all_data_1min.csv'))
    df = pd.read_csv('/kaggle/input/btcusd-test/BTCUSD_ohlcv.csv')
    # df = pd.read_csv('/kaggle/input/btcusd-test/BTC-ETH-1m.csv')
    # df = pd.read_csv('/kaggle/input/btcusd-test/all_data_1min.csv')
    # df['datetime'] = pd.to_datetime(df['datetime'])
    # df = df.iloc[-10080:]
    df = df.iloc[-525600:]
    df.sort_values('datetime', ascending=True, inplace=True)
    df.reset_index(inplace=True, drop=True)
    price_data = {}
    price_data['open'] = df['open'].values
    price_data['close'] = df['close'].values
    price_data['high'] = df['high'].values
    price_data['low'] = df['low'].values
    price_data['volume'] = df['volume'].values
    return price_data

class max_fitness_numba(base_ff):
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
    def evaluate(self, ind, **kwargs):
        p = ind.phenotype
        # print("\n" + p)
        # fitness = 0
        # for trial in range(50):
        self.test_data = generate_data()
        d = {'price_data': self.test_data}
        try:
            t0 = time.time()
            exec(p, d)
            t1 = time.time()
            fitness = d['fitness']
            # fitness += len(p)
            # v = abs(m - guess)
            # if v <= 10**6:
            #     fitness += v
            # else:
            #     fitness = self.default_fitness
            #     break
            # if t1 - t0 < 10:
            #     fitness = self.default_fitness
            #     break
            # else:
            #     fitness += (t1 - t0) * 1000
        except:
            fitness = 404
            # fitness = self.default_fitness
            # break
        if os.path.exists('ge_results.csv'):
            temp_df = pd.read_csv('ge_results.csv')
            temp_df.loc[-1] = [
                # re.findall(r"buy \= np.where\((.*), 1, 0\)", p)[0],
                # re.findall(r"sell \= np.where\((.*), -1, 0\)", p)[0],
                re.findall(r"trading_signals\(buy_signal\=(.*)\, sell_signal", p)[0],
                re.findall(r"\, sell_signal\=(.*)\)", p)[0],
                fitness
            ]
            temp_df.index = temp_df.index + 1
            temp_df = temp_df.sort_index()
            temp_df.to_csv('ge_results.csv', index=False)
        else:
            temp_dict = {}
            # temp_dict['buy'] = [re.findall(r"buy \= np.where\((.*), 1, 0\)", p)[0]]
            # temp_dict['sell'] = [re.findall(r"sell \= np.where\((.*), -1, 0\)", p)[0]]
            temp_dict['buy'] = [re.findall(r"trading_signals\(buy_signal\=(.*)\, sell_signal", p)[0]]
            temp_dict['sell'] = [re.findall(r"\, sell_signal\=(.*)\)", p)[0]]
            temp_dict['fitness'] = fitness
            temp_df = pd.DataFrame(temp_dict)
            temp_df.to_csv('ge_results.csv', index=False)
        return fitness