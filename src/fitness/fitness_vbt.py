from fitness.base_ff_classes.base_ff import base_ff
import pandas as pd
from pathlib import Path

def generate_data():

    df = pd.read_csv(Path(r'C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\Upwork/\AlgoT_ML_Dev/\GrammarEvolution/\PonyGE2/\btc_9folds_60min/\data_fold2.csv'))  
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.sort_values('datetime', ascending=True, inplace=True)
    df.reset_index(inplace=True, drop=True)

    price_data = {}
    
    for col in df.columns:
        price_data[col] = df[col].values

    price_data['day_of_week'] = (df['datetime'].dt.dayofweek + 1).values
    price_data['month'] = df['datetime'].dt.month.values
    price_data['hour'] = df['datetime'].dt.hour.values
    price_data['minute'] = df['datetime'].dt.minute.values
    
    return price_data

class fitness_vbt(base_ff):

    def __init__(self):

        super().__init__()

    def evaluate(self, ind, **kwargs):
        
        p = ind.phenotype

        self.test_data = generate_data()
        d = {'price_data': self.test_data}

        try:
            exec(p, d)
            fitness = d['fitness']
        except Exception as e:
            fitness = 404
            
        return fitness