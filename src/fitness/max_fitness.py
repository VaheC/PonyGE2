from fitness.base_ff_classes.base_ff import base_ff
import random
import time
import pandas as pd
from fitness.indicators.indicators import *

def generate_data():
    df = pd.read_csv('/content/BTCUSD_ohlcv.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.iloc[:10080]
    df.sort_values('datetime', ascending=True, inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df

class max_fitness(base_ff):
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
    def evaluate(self, ind, **kwargs):
        p = ind.phenotype
        print("\n" + p)
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
            fitness = self.default_fitness
            # break
        return fitness