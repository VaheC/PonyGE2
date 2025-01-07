from pathlib import Path
import pandas as pd

from fitness.performance import filter_strategies, portfolio_testing
from fitness.custom_logger.load_logger import create_file_logger

# set the path to the training data, used to get strategies, below
data_path = Path(r'C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\Upwork/\AlgoT_ML_Dev/\GrammarEvolution/\PonyGE2/\data_120min.csv')

# set the path to the csv file containing the strategies generated
strategy_file_path = Path(r"C:/\Users/\vchar/\Downloads/\ge_results.csv")

stats_path = 'testing_results' # folder which will contain csv files with testing stats as in testing_strategies.ipynb
file_name = 'fold1_baseline' # suffix which will be added to some csv files created during this run
str_path = 'selected_strategies' # folder which will contain csv files with strategies survived after testing
selected_str_file_path = f'{str_path}/selected_strategies_{file_name}.csv' # file path which will contain the survived strategies
lstr_path = 'live_strategies' # folder which will contain csv files with strategies survived after out of fold/sample testing
lstr_file_name='fold1_baseline' # suffix which will be added to the csv files containing the strategies survived after out of fold/sample teting
port_path = 'portfolio_strategies' # folder which will contain csv files with the portfolio weights of the survived strategies
port_file_path = f'{port_path}/portfolio_{file_name}.csv' # path to a file containing the portfolio weigths of the survived strategies
port_perf_path = 'portfolio_out_sample_performance' # folder which will contain csv files with the performance (ROI (%), PNL) of the portfolio for different out of fold/sample data
str_perf_path='str_out_sample_performance' # folder which will contain csv files with the performance (ROI (%)) of the survived strategies for different out of fold/sample data

# initialize the log file   
logger = create_file_logger(filename=f'{file_name}')

# set the time frequency in minutes
time_freq = 120

# calculate the number of bars per 5 weeks
# n_bars = int(7 * 24 * 60 * 5 / time_freq) # use in case of 1, 5, and 30 min frequencies
n_bars = int(7 * 24 * 60 * 40 / time_freq) # use in case of 120 and 240 min frequencies
# n_bars = int(7 * 24 * 60 * 10 / time_freq) # use in case of 60 min frequency

# set the number of available folds
# n_total_folds = 2

try:
    df = pd.read_csv(data_path)
except:
    df = pd.read_csv(data_path, sep=';')
n_total_folds = int((df.shape[0] - (df.shape[0]  % n_bars)) / n_bars)
if df.shape[0]  % n_bars != 0:
    n_total_folds += 1
del df

# n_fold should be equal to the number of fold which has been used to derive the strategies
# if there are no folds, just set it to 0
n_fold = 1

# start_fold should be set to the number of fold starting from which the folds of data are not used during strategy generation
# end_fold is the last fold not used for strategy generation
start_fold = 1
end_fold = n_total_folds

# setting winning probability threshold to filter strategies
prob_threshold = 0.98

# setting investment amount
initial_capital = 700000
trade_size = 0.5
initial_amount = initial_capital * trade_size

# acquiring the stats for the generated strategies
# the function will create testing_results folder inside PonyGE2\src to save the results
filter_strategies.save_stats(
    data_path=data_path, 
    strategy_file_path=strategy_file_path, 
    stats_file_name=file_name,
    n_fold=n_fold, 
    logger=logger, 
    n_bars=n_bars,
    # create_txt_code = filter_strategies.create_txt_code1_vbt # use if you want to run the test using VectorBT
)

# filtering the strategies based on the limiting test
# the function will create selected_strategies folder inside PonyGE2\src to save the results
filter_strategies.filter_save_strategies(
    strategy_file_path, 
    logger, 
    stats_path=stats_path, 
    stats_file_name=file_name, 
    str_path=str_path, 
    str_file_name=file_name, 
    entry_testing_threshold=60, # winning percentage threshold for entry testing
    exit_testing_threshold=60, # winning percentage threshold for exit testing
    core_testing_threshold=70, # winning percentage threshold for core testing
    prob_threshold=prob_threshold,
    is_counter_trend_exit=False, # set to True if you want to include counter trend in exit testing
    is_random_exit=False # set to True if you want to include random exit in entry testing
)

# filtering the strategies based on the limiting test and out of sample data (all folds except n_fold)
# the function will create live_strategies folder inside PonyGE2\src to save the results
portfolio_testing.filter_save_lstr(
    data_path, 
    n_fold=n_fold, 
    str_file_path=selected_str_file_path, 
    logger=logger,
    lstr_path=lstr_path, 
    lstr_file_name=lstr_file_name, 
    # is_subset=False, # skip
    # start_subset=256, # skip
    # end_subset=260, # skip
    entry_testing_threshold=50, # winning percentage threshold for entry testing
    exit_testing_threshold=50, # winning percentage threshold for exit testing
    core_testing_threshold=60, # winning percentage threshold for core testing
    prob_threshold=prob_threshold, 
    n_bars=n_bars, 
    n_total_folds=n_total_folds,
    is_counter_trend_exit=False, # set to True if you want to include counter trend in exit testing
    is_random_exit=False, # set to True if you want to include random exit in entry testing
    entry_exit_on=False, # set to True if you want to apply exit and entry testing, usually I use only core testing for out of sample data
    # create_txt_code=portfolio_testing.create_txt_code1_vbt # use if you want to run the test using VectorBT
)

# creating portfolio weights using prob
# the function will create portfolio_strategies folder inside PonyGE2\src to save the results
portfolio_testing.creating_port_weights(
    lstr_path, 
    logger, 
    port_path=port_path, 
    port_file_name=file_name, 
    is_prob=False, 
    prob_threshold=prob_threshold
)

# calculating out of sample (some folds that are not touched during strategy derivation) performance of the portfolio derived above
# the function will create portfolio_out_sample_performance folder inside PonyGE2\src to save the results
portfolio_testing.calculate_port_out_sample_perf(
    data_path, 
    port_file_path, 
    logger, 
    n_bars=n_bars, 
    initial_amount=initial_amount, 
    start_fold=start_fold, 
    end_fold=end_fold, 
    port_perf_path=port_perf_path,
    port_perf_file_name=file_name,
    # create_txt_code_port=portfolio_testing.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# creating portfolio weights using prob and setting threshold for prob
# the function will save the results inside portfolio_strategies in PonyGE2\src
portfolio_testing.creating_port_weights(
    lstr_path, 
    logger, 
    port_path=port_path, 
    port_file_name=file_name, 
    is_prob=True, 
    prob_threshold=prob_threshold
)

# calculating out of sample (some folds that are not touched during strategy derivation) performance of the portfolio derived above
# the function will save the results inside portfolio_out_sample_performance in PonyGE2\src
portfolio_testing.calculate_port_out_sample_perf(
    data_path, 
    port_file_path, 
    logger, 
    n_bars=n_bars, 
    initial_amount=initial_amount, 
    start_fold=start_fold, 
    end_fold=end_fold, 
    port_perf_path=port_perf_path,
    port_perf_file_name=f'{file_name}_prob',
    # create_txt_code_port=portfolio_testing.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# creating portfolio weights by maximizing Sortino ratio
# the function will save the results inside portfolio_strategies in PonyGE2\src
portfolio_testing.creating_port_weights_mvp(
    lstr_path, 
    data_path, 
    port_path=port_path, 
    port_file_name=file_name.replace('_baseline', ''),
    is_min_variance_port=False, # True when min_variance is used to derive weights
    is_sharpe_port=False, # True when Sharpe ratio is used to derive weights
    is_sortino_port=True, # True when Sortino ratio is used to derive weights
    prob_threshold=prob_threshold,
    n_bars=n_bars, 
    n_total_folds=n_total_folds, 
    freq_minutes=time_freq,
    # create_txt_code=portfolio_testing.create_txt_code_pnl1_vbt # use if you want to run the test using VectorBT
)

# calculating out of sample (some folds that are not touched during strategy derivation) performance of the portfolio derived above
# the function will save the results inside portfolio_out_sample_performance in PonyGE2\src
portfolio_testing.calculate_port_out_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name.replace('_baseline', '')}_sortino.csv", 
    logger=logger, 
    n_bars=n_bars, 
    initial_amount=initial_amount, 
    start_fold=start_fold, 
    end_fold=end_fold, 
    port_perf_path=port_perf_path,
    port_perf_file_name='sortino',
    # create_txt_code_port=portfolio_testing.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# creating portfolio weights by maximizing Sharpe ratio
# the function will save the results inside portfolio_strategies in PonyGE2\src
portfolio_testing.creating_port_weights_mvp(
    lstr_path, 
    data_path, 
    port_path=port_path, 
    port_file_name=file_name.replace('_baseline', ''),
    is_min_variance_port=False, # True when min_variance is used to derive weights
    is_sharpe_port=True, # True when Sharpe ratio is used to derive weights
    is_sortino_port=False, # True when Sortino ratio is used to derive weights
    prob_threshold=prob_threshold,
    n_bars=n_bars, 
    n_total_folds=n_total_folds, 
    freq_minutes=time_freq,
    # create_txt_code=portfolio_testing.create_txt_code_pnl1_vbt # use if you want to run the test using VectorBT
)

# calculating out of sample (some folds that are not touched during strategy derivation) performance of the portfolio derived above
# the function will save the results inside portfolio_out_sample_performance in PonyGE2\src
portfolio_testing.calculate_port_out_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name.replace('_baseline', '')}_sharpe.csv", 
    logger=logger, 
    n_bars=n_bars, 
    initial_amount=initial_amount, 
    start_fold=start_fold, 
    end_fold=end_fold, 
    port_perf_path=port_perf_path,
    port_perf_file_name='sharpe',
    # create_txt_code_port=portfolio_testing.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# creating portfolio weights by minimizing variance/risk
# the function will save the results inside portfolio_strategies in PonyGE2\src
portfolio_testing.creating_port_weights_mvp(
    lstr_path, 
    data_path, 
    port_path=port_path, 
    port_file_name=file_name.replace('_baseline', ''),
    is_min_variance_port=True, # True when min_variance is used to derive weights
    is_sharpe_port=False, # True when Sharpe ratio is used to derive weights
    is_sortino_port=False, # True when Sortino ratio is used to derive weights
    prob_threshold=prob_threshold,
    n_bars=n_bars, 
    n_total_folds=n_total_folds, 
    freq_minutes=time_freq,
    # create_txt_code=portfolio_testing.create_txt_code_pnl1_vbt # use if you want to run the test using VectorBT
)

# calculating out of sample (some folds that are not touched during strategy derivation) performance of the portfolio derived above
# the function will save the results inside portfolio_out_sample_performance in PonyGE2\src
portfolio_testing.calculate_port_out_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name.replace('_baseline', '')}_mvp.csv", 
    logger=logger, 
    n_bars=n_bars, 
    initial_amount=initial_amount, 
    start_fold=start_fold, 
    end_fold=end_fold, 
    port_perf_path=port_perf_path,
    port_perf_file_name='mvp',
    # create_txt_code_port=portfolio_testing.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# creating portfolio weights using hierarchical risk parity
# the function will save the results inside portfolio_strategies in PonyGE2\src
portfolio_testing.creating_port_weights_hrp(
    lstr_path, 
    data_path, 
    n_bars=n_bars, 
    n_total_folds=n_total_folds, 
    port_path=port_path, 
    port_file_name=file_name.replace('_baseline', '_hrp'),
    is_prob=True, 
    prob_threshold=prob_threshold,
    # create_txt_code=portfolio_testing.create_txt_code_pnl1_vbt # use if you want to run the test using VectorBT
)

# calculating out of sample (some folds that are not touched during strategy derivation) performance of the portfolio derived above
# the function will save the results inside portfolio_out_sample_performance in PonyGE2\src
portfolio_testing.calculate_port_out_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name.replace('_baseline', '')}_hrp.csv", 
    logger=logger, 
    n_bars=n_bars, 
    initial_amount=initial_amount, 
    start_fold=start_fold, 
    end_fold=end_fold, 
    port_perf_path=port_perf_path,
    port_perf_file_name='hrp',
    # create_txt_code_port=portfolio_testing.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# calculating out of sample (some folds that are not touched during strategy derivation) perfomance of all survived strategies
# the function will create str_out_sample_performance folder inside PonyGE2\src to save the results
portfolio_testing.calculate_str_out_sample_perf(
    data_path, 
    port_file_path, 
    logger, 
    n_bars=n_bars, 
    initial_amount=initial_amount, 
    start_fold=start_fold, 
    end_fold=end_fold, 
    str_perf_path=str_perf_path,
    str_perf_file_name='str_roi',
    # create_txt_code_port=portfolio_testing.create_txt_code1_vbt # use if you want to run the test using VectorBT
)
