from pathlib import Path

from fitness.performance import filter_strategies, portfolio_testing
from fitness.custom_logger.load_logger import create_file_logger

# set the path to the training data, used to get strategies, below
data_path = Path(r'C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\Upwork/\AlgoT_ML_Dev/\GrammarEvolution/\PonyGE2/\all_data_1min_all.csv')

# set the path to the csv file containing the strategies generated
strategy_file_path = Path(r"C:/\Users/\vchar/\Downloads/\ge_results (6).csv")

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
time_freq = 60

# calculate the number of bars per 5 weeks
n_bars = int(7 * 24 * 60 * 5 / time_freq)

# set the number of available folds
n_total_folds = 9

# n_fold should be equal to the number of fold which has been used to derive the strategies
# if there are no folds, just set it to 0
n_fold = 0

# start_fold should be set to the number of fold starting from which the folds of data are not used during strategy generation
# end_fold is the last fold not used for strategy generation
start_fold=1
end_fold=9

# acquiring the stats for the generated strategies
filter_strategies.save_stats(
    data_path=data_path, strategy_file_path=strategy_file_path, 
    n_fold=n_fold, logger=logger, n_bars=n_bars
)

# filtering the strategies based on the limiting test
filter_strategies.filter_save_strategies(strategy_file_path, logger, 
    stats_path=stats_path, stats_file_name=file_name, str_path=str_path, 
    str_file_name=file_name, entry_testing_threshold=70,
    exit_testing_threshold=60, core_testing_threshold=70,
    prob_threshold=0.98
)

# filtering the strategies based on the limiting test and out of sample data (all folds except n_fold)
portfolio_testing.filter_save_lstr(data_path, n_fold=n_fold, str_file_path=selected_str_file_path, logger=logger,
    lstr_path=lstr_path, lstr_file_name=lstr_file_name, is_subset=False, start_subset=256, end_subset=260,
    entry_testing_threshold=50, exit_testing_threshold=50, core_testing_threshold=60,
    prob_threshold=0.8, n_bars=n_bars, n_total_folds=n_total_folds
)

# creating portfolio weights using prob
portfolio_testing.creating_port_weights(lstr_path, logger, port_path=port_path, 
    port_file_name=file_name, is_prob=False, prob_threshold=0.98
)

# calculating out of sample (some folds that are not touched during strategy derivation) performance of the portfolio derived above
portfolio_testing.calculate_port_out_sample_perf(
    data_path, port_file_path, logger, n_bars=n_bars, 
    initial_amount=350000, start_fold=start_fold, end_fold=end_fold, 
    port_perf_path=port_perf_path,
    port_perf_file_name=file_name
)

# creating portfolio weights using prob and setting threshold for prob
portfolio_testing.creating_port_weights(lstr_path, logger, port_path=port_path, 
    port_file_name=file_name, is_prob=True, prob_threshold=0.98
)

# calculating out of sample (some folds that are not touched during strategy derivation) performance of the portfolio derived above
portfolio_testing.calculate_port_out_sample_perf(
    data_path, port_file_path, logger, n_bars=n_bars, 
    initial_amount=350000, start_fold=start_fold, end_fold=end_fold, 
    port_perf_path=port_perf_path,
    port_perf_file_name=f'{file_name}_prob'
)

# creating portfolio weights by maximizing Sortino ratio
portfolio_testing.creating_port_weights_mvp(
    lstr_path, data_path, port_path=port_path, port_file_name=file_name.replace('_baseline', ''),
    is_min_variance_port=False, is_sharpe_port=False, is_sortino_port=True, prob_threshold=0.98,
    n_bars=n_bars, n_total_folds=n_total_folds, freq_minutes=time_freq
)

# calculating out of sample (some folds that are not touched during strategy derivation) performance of the portfolio derived above
portfolio_testing.calculate_port_out_sample_perf(
    data_path, port_file_path=f'{port_path}/{file_name.replace('_baseline', '')}_sortino.csv', 
    logger=logger, n_bars=n_bars, 
    initial_amount=350000, start_fold=start_fold, end_fold=end_fold, 
    port_perf_path=port_perf_path,
    port_perf_file_name='sortino'
)

# creating portfolio weights by maximizing Sharpe ratio
portfolio_testing.creating_port_weights_mvp(
    lstr_path, data_path, port_path=port_path, port_file_name=file_name.replace('_baseline', ''),
    is_min_variance_port=False, is_sharpe_port=True, is_sortino_port=False, prob_threshold=0.98,
    n_bars=n_bars, n_total_folds=n_total_folds, freq_minutes=time_freq
)

# calculating out of sample (some folds that are not touched during strategy derivation) performance of the portfolio derived above
portfolio_testing.calculate_port_out_sample_perf(
    data_path, port_file_path=f'{port_path}/{file_name.replace('_baseline', '')}_sharpe.csv', 
    logger=logger, n_bars=n_bars, 
    initial_amount=350000, start_fold=start_fold, end_fold=end_fold, 
    port_perf_path=port_perf_path,
    port_perf_file_name='sharpe'
)

# creating portfolio weights by minimizing variance/risk
portfolio_testing.creating_port_weights_mvp(
    lstr_path, data_path, port_path=port_path, port_file_name=file_name.replace('_baseline', ''),
    is_min_variance_port=True, is_sharpe_port=False, is_sortino_port=False, prob_threshold=0.98,
    n_bars=n_bars, n_total_folds=n_total_folds, freq_minutes=60)

# calculating out of sample (some folds that are not touched during strategy derivation) performance of the portfolio derived above
portfolio_testing.calculate_port_out_sample_perf(
    data_path, port_file_path=f'{port_path}/{file_name.replace('_baseline', '')}_mvp.csv', 
    logger=logger, n_bars=n_bars, 
    initial_amount=350000, start_fold=start_fold, end_fold=end_fold, 
    port_perf_path=port_perf_path,
    port_perf_file_name='mvp'
)

# creating portfolio weights using hierarchical risk parity
portfolio_testing.creating_port_weights_hrp(
    lstr_path, data_path, n_bars=n_bars, 
    n_total_folds=n_total_folds, 
    port_path=port_path, 
    port_file_name=file_name.replace('_baseline', '_hrp'),
    is_prob=True, prob_threshold=0.98
)

# calculating out of sample (some folds that are not touched during strategy derivation) performance of the portfolio derived above
portfolio_testing.calculate_port_out_sample_perf(
    data_path, port_file_path=f'{port_path}/{file_name.replace('_baseline', '')}_hrp.csv', 
    logger=logger, n_bars=n_bars, 
    initial_amount=350000, start_fold=start_fold, end_fold=end_fold, 
    port_perf_path=port_perf_path,
    port_perf_file_name='hrp'
)

# calculating out of sample (some folds that are not touched during strategy derivation) perfomance of all survived strategies
portfolio_testing.calculate_str_out_sample_perf(
    data_path, port_file_path, logger, n_bars=n_bars, 
    initial_amount=350000, start_fold=start_fold, end_fold=end_fold, 
    str_perf_path=str_perf_path,
    str_perf_file_name='str_roi'
)