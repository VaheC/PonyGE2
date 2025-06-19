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
    
    df = pd.read_csv('/kaggle/input/wq-dataset/wq_train_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(['date', 'stock'], ascending=True, inplace=True)
    df.reset_index(inplace=True, drop=True)

    price_data = {}
   
    for col in df.columns:
        if col in ['date', 'stock']:
            continue
        else:
            price_data[col] = df.pivot(index='date', columns='stock', values=col).values
    return price_data

class fitness_wq_kaggle(base_ff):

    def __init__(self):

        super().__init__()

    def evaluate(self, ind, **kwargs):

        p = ind.phenotype

        self.test_data = generate_data()
        d = {'price_data': self.test_data}

        # logger = create_terminal_logger()

        # try:
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
        # except:
        #     fitness = 404
            
        return fitness