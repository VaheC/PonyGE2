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
    df = pd.read_csv('/kaggle/input/btcusd-test/BTCUSD_ohlcv.csv')
    # df = pd.read_csv('/kaggle/input/btcusd-test/BTC-ETH-1m.csv')
    # df = pd.read_csv('/kaggle/input/btcusd-test/all_data_1min.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.iloc[-10080:]
    df.sort_values('datetime', ascending=True, inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df

class max_fitness(base_ff):
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
    def evaluate(self, ind, **kwargs):
        p = ind.phenotype
        # print("\n" + p)
        # fitness = 0
        # for trial in range(50):
        self.test_data = generate_data()
        d = {'data': self.test_data}
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
                re.findall(r"df\[\'buy\'\] = \((.*)\)\.astype\(int\)", p)[0],
                re.findall(r"df\[\'sell\'\] = \((.*)\)\.astype\(int\)", p)[0],
                fitness
            ]
            temp_df.index = temp_df.index + 1
            temp_df = temp_df.sort_index()
            temp_df.to_csv('ge_results.csv', index=False)
        else:
            temp_dict = {}
            temp_dict['buy'] = [re.findall(r"df\[\'buy\'\] = \((.*)\)\.astype\(int\)", p)[0]]
            temp_dict['sell'] = [re.findall(r"df\[\'sell\'\] = \((.*)\)\.astype\(int\)", p)[0]]
            temp_dict['fitness'] = fitness
            temp_df = pd.DataFrame(temp_dict)
            temp_df.to_csv('ge_results.csv', index=False)
        return fitness