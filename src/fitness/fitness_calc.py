from fitness.base_ff_classes.base_ff import base_ff
from pathlib import Path
import polars as pl
import re

def generate_data():
    df = pl.read_csv(Path(r'/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\Upwork/\AlgoT_ML_Dev/\GrammarEvolution/\PonyGE2/\all_data_1min.csv'))

    df = df.with_columns(pl.col('datetime').str.to_datetime())
    df = df.sort('datetime', descending=False)
    df = df.slice(252000, 50400) # 7 * 60 * 24 * 5 = 50400, 7 * 60 * 24 * 25 = 252000

    price_data = {}

    for col in df.columns:
        if col == 'datetime':
            continue
        else:
            price_data[col] = df.get_column(col).to_numpy()
    price_data['day_of_week'] = df.get_column('datetime').dt.weekday().to_numpy()
    return price_data

class fitness_calc(base_ff):

    def __init__(self):

        super().__init__()

    def evaluate(self, ind, **kwargs):

        p = ind.phenotype

        # print("\n" + p)

        self.test_data = generate_data()
        d = {'price_data': self.test_data}

        try:
            # t0 = time.time()
            exec(p, d)
            # t1 = time.time()
            fitness = d['fitness']
        except:
            fitness = 404

        # buy_signal = re.findall(r"trading_signals_buy\(buy_signal\=(.*)\, exit_signal", p)[0]
        # buy_exit_signal = re.findall(r"\, exit_signal\=(.*)\)", p)[0]

        # print('#'*40)

        # print(f"BUY signal: {buy_signal}")
        # print(f"BUY Exit signal: {buy_exit_signal}")

        # sell_signal = re.findall(r"trading_signals_sell\(sell_signal\=(.*)\, exit_signal", p)[0]
        # sell_exit_signal = re.findall(r"\, exit_signal\=(.*)\)", p)[0]

        # print(f"SELL signal: {sell_signal}")
        # print(f"SELL Exit signal: {sell_exit_signal}")
        # print('#'*40)
            
        return fitness