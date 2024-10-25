import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import timedelta, datetime

import warnings
warnings.filterwarnings('ignore')

main_path = Path(r'C:/\Users/\vchar/\Downloads/\FULL DATA LIBRARY')

files_list = os.listdir(main_path)

final_df = pd.read_csv(os.path.join(main_path, files_list[4]))
instrument_name = files_list[4].replace('-', '_').split('_')[0].lower()
final_df['datetime'] = pd.to_datetime(final_df['datetime'])
final_df.sort_values('datetime', ascending=True, inplace=True)
final_df.reset_index(inplace=True, drop=True)
final_df.rename(
    columns={
        'open': f'{instrument_name}_open', 
        'close': f'{instrument_name}_close', 
        'low': f'{instrument_name}_low', 
        'high': f'{instrument_name}_high', 
        'volume': f'{instrument_name}_volume'
    },
    inplace=True
)

final_df.set_index('datetime', inplace=True)
complete_time_index = pd.date_range(start=final_df.index.min(), end=final_df.index.max(), freq='min')
final_df = final_df.reindex(complete_time_index)
final_df = final_df.ffill()
final_df.reset_index(inplace=True)
final_df.rename(columns={'index': 'datetime'}, inplace=True)

for i in range(len(files_list)):

    if i in [4, 9, 21]:

        if i != 4:
            print(f"{files_list[i]} doesn't contain overlapping data.")
        continue

    temp_df = pd.read_csv(os.path.join(main_path, files_list[i]))
    instrument_name = files_list[i].replace('-', '_').split('_')[0].lower()
    temp_df['datetime'] = pd.to_datetime(temp_df['datetime'])
    temp_df.sort_values('datetime', ascending=True, inplace=True)
    temp_df.reset_index(inplace=True, drop=True)
    temp_df.rename(
        columns={
            'open': f'{instrument_name}_open', 
            'close': f'{instrument_name}_close', 
            'low': f'{instrument_name}_low', 
            'high': f'{instrument_name}_high', 
            'volume': f'{instrument_name}_volume'
        },
        inplace=True
    )
    
    final_df = pd.merge(final_df, temp_df, on='datetime', how='left')
    final_df.ffill(inplace=True)

    if final_df.shape[0] == 0:
        print(i)

final_df.dropna(inplace=True)
final_df.reset_index(drop=True, inplace=True)

final_df.to_csv("all_data_1min.csv", index=False)

df = pd.read_csv('all_data_1min.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df.sort_values('datetime', ascending=True, inplace=True)
df.reset_index(inplace=True, drop=True)

instrument_list = [col.split('_')[0] for col in df.columns if col.endswith('_open')]

print(df.shape)
df.set_index('datetime', inplace=True)

# Grouping by date
df['date'] = df.index.date
grouped = df.groupby('date')

df['weekend_date'] = df.index
df['weekend_date'] = df['weekend_date'].apply(lambda x: (x + timedelta(days=6 - x.weekday())).date())

df['month_date'] = df.index
df['month_date'] = df['month_date'].apply(
    lambda x: (
        datetime(x.year + int(x.month % 12 == 0), x.month % 12 + 1, 1) - timedelta(days=1)
    ).date()
)

for instrument in instrument_list:

    daily_data = df.resample('1D').agg({
        f'{instrument}_open': 'first',
        f'{instrument}_high': 'max',
        f'{instrument}_low': 'min',
        f'{instrument}_close': 'last',
        f'{instrument}_volume': 'sum'
    })

    weekly_data = df.resample('W').agg({
        f'{instrument}_open': 'first',
        f'{instrument}_high': 'max',
        f'{instrument}_low': 'min',
        f'{instrument}_close': 'last',
        f'{instrument}_volume': 'sum'
    })

    monthly_data = df.resample('ME').agg({
        f'{instrument}_open': 'first',
        f'{instrument}_high': 'max',
        f'{instrument}_low': 'min',
        f'{instrument}_close': 'last',
        f'{instrument}_volume': 'sum'
    })

    df[f'{instrument}_mean'] = (df[f'{instrument}_close'] + df[f'{instrument}_low'] + df[f'{instrument}_high'] + df[f'{instrument}_open']) / 4

    df[f'{instrument}_rmean_day'] = grouped[f'{instrument}_close'].transform(lambda x: x.shift(1).rolling(window=len(x), min_periods=1).mean())
    df[f'{instrument}_rhigh_day'] = grouped[f'{instrument}_high'].transform(lambda x: x.shift(1).rolling(window=len(x), min_periods=1).max())
    df[f'{instrument}_rlow_day'] = grouped[f'{instrument}_low'].transform(lambda x: x.shift(1).rolling(window=len(x), min_periods=1).min())
    df[f'{instrument}_rstd_day'] = grouped[f'{instrument}_close'].transform(lambda x: x.shift(1).rolling(window=len(x), min_periods=1).std())
    df[f'{instrument}_rvolume_day'] = grouped[f'{instrument}_volume'].transform(lambda x: x.shift(1).rolling(window=len(x), min_periods=1).sum())

    df[f'{instrument}_rmean_2h'] = df[f'{instrument}_close'].shift(1).rolling(window=120, min_periods=1).mean()
    df[f'{instrument}_rmax_2h'] = df[f'{instrument}_high'].shift(1).rolling(window=120, min_periods=1).max()
    df[f'{instrument}_rmin_2h'] = df[f'{instrument}_low'].shift(1).rolling(window=120, min_periods=1).min()
    df[f'{instrument}_rstd_2h'] = df[f'{instrument}_close'].shift(1).rolling(window=120, min_periods=1).std()
    df[f'{instrument}_rvolume_2h'] = df[f'{instrument}_volume'].shift(1).rolling(window=120, min_periods=1).sum()

    df[f'{instrument}_rmean_4h'] = df[f'{instrument}_close'].shift(1).rolling(window=240, min_periods=1).mean()
    df[f'{instrument}_rmax_4h'] = df[f'{instrument}_high'].shift(1).rolling(window=240, min_periods=1).max()
    df[f'{instrument}_rmin_4h'] = df[f'{instrument}_low'].shift(1).rolling(window=240, min_periods=1).min()
    df[f'{instrument}_rstd_4h'] = df[f'{instrument}_close'].shift(1).rolling(window=240, min_periods=1).std()
    df[f'{instrument}_rvolume_4h'] = df[f'{instrument}_volume'].shift(1).rolling(window=240, min_periods=1).sum()

    df[f'{instrument}_rmean_1h'] = df[f'{instrument}_close'].shift(1).rolling(window=60, min_periods=1).mean()
    df[f'{instrument}_rmax_1h'] = df[f'{instrument}_high'].shift(1).rolling(window=60, min_periods=1).max()
    df[f'{instrument}_rmin_1h'] = df[f'{instrument}_low'].shift(1).rolling(window=60, min_periods=1).min()
    df[f'{instrument}_rstd_1h'] = df[f'{instrument}_close'].shift(1).rolling(window=60, min_periods=1).std()
    df[f'{instrument}_rvolume_1h'] = df[f'{instrument}_volume'].shift(1).rolling(window=60, min_periods=1).sum()
    
    daily_cols = []
    weekly_cols = []
    monthly_cols = []

    for col in ['high', 'low', 'open', 'close', 'volume']:

        for n_lag in [1, 3, 5]:
            daily_data[f'{instrument}_{col}_{n_lag}d_ago'] = daily_data[f'{instrument}_{col}'].shift(n_lag)
            daily_cols.append(f'{instrument}_{col}_{n_lag}d_ago')

        weekly_data[f'{instrument}_{col}_1w_ago'] = weekly_data[f'{instrument}_{col}'].shift(1)
        weekly_cols.append(f'{instrument}_{col}_1w_ago')

        monthly_data[f'{instrument}_{col}_1m_ago'] = monthly_data[f'{instrument}_{col}'].shift(1)
        monthly_cols.append(f'{instrument}_{col}_1m_ago')

    daily_data = daily_data[daily_cols].reset_index()
    weekly_data = weekly_data[weekly_cols].reset_index()
    monthly_data = monthly_data[monthly_cols].reset_index()

    daily_map_dict = daily_data.set_index('datetime').to_dict()
    for k in daily_map_dict.keys():
        df[k] = df['date'].map(daily_map_dict[k])

    weekly_map_dict = weekly_data.set_index('datetime').to_dict()
    for k in weekly_map_dict.keys():
        df[k] = df['weekend_date'].map(weekly_map_dict[k])

    monthly_map_dict = monthly_data.set_index('datetime').to_dict()
    for k in monthly_map_dict.keys():
        df[k] = df['month_date'].map(monthly_map_dict[k])

df.drop(columns=['date', 'weekend_date', 'month_date'], inplace=True)
df.dropna(inplace=True)
df.reset_index(inplace=True)
print(df.shape)

df.to_csv("all_data_1min.csv", index=False)