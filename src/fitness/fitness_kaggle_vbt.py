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

    # df = pd.read_csv('/kaggle/input/btcusd-test/data_folds_60min/data_fold3.csv')
    # df = pd.read_csv('/kaggle/input/btcusd-test/data_60min_train.csv')
    # df = pd.read_csv('/kaggle/input/btcusd-test/data_folds_30min/data_fold4.csv')
    # df = pd.read_csv('/kaggle/input/btcusd-test/data_folds_120min/data_fold1.csv')
    df = pd.read_csv('/kaggle/input/btcusd-test/btc_10folds_120min/data_fold6.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])

    # df['year'] = df['datetime'].dt.year
    # random_year = list(df['year'].unique())[1]
    # df = df[df['year'] == random_year]
    # df.drop(columns=['year'], inplace=True)

    df.sort_values('datetime', ascending=True, inplace=True)
    df.reset_index(inplace=True, drop=True)

    price_data = {}
   
    for col in df.columns:
        price_data[col] = df[col].values.reshape(-1, 1)

    price_data['day_of_week'] = (df['datetime'].dt.dayofweek + 1).values.reshape(-1, 1)
    price_data['month'] = df['datetime'].dt.month.values.reshape(-1, 1)
    price_data['hour'] = df['datetime'].dt.hour.values.reshape(-1, 1)
    price_data['minute'] = df['datetime'].dt.minute.values.reshape(-1, 1)
    # price_data['month'] = df['datetime'].dt.month.values
    # price_data['day_of_year'] = df['datetime'].dt.dayofyear.values
    return price_data

class fitness_kaggle_vbt(base_ff):

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

            # temp_txt_logger = f"ROI: {roi:.4f}, MDD: {mdd:.4f}, Fitness: {fitness:.4f}"

            # logger.info(temp_txt_logger)
        except:
            fitness = 404
            
        return fitness