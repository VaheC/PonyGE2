from pathlib import Path

from fitness.performance import filter_strategies3, portfolio_testing3
from fitness.custom_logger.load_logger import create_file_logger

# set the path to the training data, used to get strategies, below
data_path = Path(r"C:/\Users/\vchar/\Downloads/\btcusd_1-min_data.csv")

# set the path to the csv file containing the strategies generated
strategy_file_path = Path(r"C:/\Users/\vchar/\Downloads/\ge_results.csv")

stats_path = 'testing_results' # folder which will contain csv files with testing stats as in testing_strategies.ipynb
file_name = 'fold6' # suffix which will be added to some csv files created during this run
str_path = 'selected_strategies' # folder which will contain csv files with strategies survived after testing
selected_str_file_path = f'{str_path}/selected_strategies_{file_name}.csv' # file path which will contain the survived strategies
lstr_path = 'live_strategies' # folder which will contain csv files with strategies survived over validation period
lstr_file_name = 'fold6' # suffix which will be added to the csv files containing the strategies survived after testing over validation period
port_path = 'portfolio_strategies' # folder which will contain csv files with the portfolio weights of the survived strategies
port_file_path = f'{port_path}/portfolio_{file_name}.csv' # path to a file containing the portfolio weigths of the survived strategies
port_perf_path = 'portfolio_out_sample_performance' # folder which will contain csv files with the performance (ROI (%), PNL) of the portfolio over validation period
str_perf_path='str_out_sample_performance' # folder which will contain csv files with the performance (ROI (%)) of the survived strategies over validation period
port_perf_path_train = 'portfolio_in_sample_performance' # folder which will contain csv files with the performance (ROI (%), PNL) of the portfolio over training period
str_perf_path_train = 'str_in_sample_performance' # folder which will contain csv files with the performance (ROI (%)) of the survived strategies over training period
port_perf_path_test = 'portfolio_test_sample_performance' # folder which will contain csv files with the performance (ROI (%), PNL) of the portfolio over testing period
str_perf_path_test = 'str_test_sample_performance' # folder which will contain csv files with the performance (ROI (%)) of the survived strategies over testing period

# initialize the log file   
logger = create_file_logger(filename=f'{file_name}')

# set the time frequency in minutes
time_freq = 60

# set the number of weeks for each fold
fold_size = 5

# n_fold should be equal to the number of fold which has been used to derive the strategies
n_fold = 6

# year should be equal to the year for which the weekly folds are constructed
year = 2024

# setting winning probability threshold 
prob_threshold = 0.98

# setting investment amount
initial_capital = 700000
trade_size = 0.5
initial_amount = initial_capital * trade_size

# acquiring the stats for the generated strategies
# the function will create testing_results folder inside PonyGE2\src to save the results
filter_strategies3.save_stats(
    data_path=data_path, 
    strategy_file_path=strategy_file_path, 
    stats_file_name=file_name,
    n_fold=n_fold, 
    year=year,
    logger=logger, 
    fold_size=fold_size, 
    time_freq=time_freq,
    # create_txt_code=filter_strategies3.create_txt_code1_vbt # use if you want to run the test using VectorBT
)

# filtering the strategies over training period based on the limiting test 
# the function will create selected_strategies folder inside PonyGE2\src to save the results
filter_strategies3.filter_save_strategies(
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

# filtering the strategies over validation period based on the limiting test
# the function will create live_strategies folder inside PonyGE2\src to save the results
portfolio_testing3.filter_save_lstr(
    data_path, 
    n_fold=n_fold, 
    str_file_path=selected_str_file_path, 
    logger=logger,
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    lstr_path=lstr_path, 
    lstr_file_name=lstr_file_name, 
    # is_subset=False, # skip
    # start_subset=256, # skip
    # end_subset=260, # skip
    entry_testing_threshold=60, # winning percentage threshold for entry testing
    exit_testing_threshold=60, # winning percentage threshold for exit testing
    core_testing_threshold=70, # winning percentage threshold for core testing
    prob_threshold=prob_threshold, 
    is_counter_trend_exit=False, # set to True if you want to include counter trend in exit testing
    is_random_exit=False, # set to True if you want to include random exit in entry testing
    entry_exit_on=True, # set to True if you want to apply exit and entry testing, usually I use only core testing for out of sample data
    # create_txt_code=portfolio_testing3.create_txt_code1_vbt # use if you want to run the test using VectorBT
)

# creating portfolio weights using prob over training period
# the function will create portfolio_strategies folder inside PonyGE2\src to save the results
portfolio_testing3.creating_port_weights(
    lstr_path, 
    logger=logger,
    lstr_file_name=lstr_file_name, 
    port_path=port_path, 
    port_file_name=file_name, 
    is_prob=False, 
    prob_threshold=prob_threshold
)

# calculating performance of the portfolio derived above over training period
# the function will create portfolio_in_sample_performance folder inside PonyGE2\src to save the results
filter_strategies3.calculate_port_in_sample_perf(
    data_path, 
    port_file_path, 
    logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq, 
    year=year,
    initial_amount=initial_amount, 
    port_perf_path=port_perf_path_train,
    port_perf_file_name=file_name,
    # create_txt_code_port=filter_strategies3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over validation period
# the function will create portfolio_out_sample_performance folder inside PonyGE2\src to save the results
portfolio_testing3.calculate_port_out_sample_perf(
    data_path, 
    port_file_path, 
    logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq, 
    year=year,
    initial_amount=initial_amount, 
    port_perf_path=port_perf_path,
    port_perf_file_name=file_name,
    # create_txt_code_port=portfolio_testing3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over test period
# the function will create portfolio_test_sample_performance folder inside PonyGE2\src to save the results
portfolio_testing3.calculate_port_test_sample_perf(
    data_path, 
    port_file_path, 
    logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq, 
    year=year,
    initial_amount=initial_amount, 
    port_perf_path=port_perf_path_test,
    port_perf_file_name=file_name,
    # create_txt_code_port=portfolio_testing3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# creating portfolio weights using prob and setting threshold for prob over training period
# the function will save the results inside portfolio_strategies in PonyGE2\src
portfolio_testing3.creating_port_weights(
    lstr_path, 
    logger=logger,
    lstr_file_name=lstr_file_name,
    port_path=port_path, 
    port_file_name=file_name, 
    is_prob=True, 
    prob_threshold=prob_threshold
)

# calculating performance of the portfolio derived above over training period
# the function will save the results inside portfolio_in_sample_performance in PonyGE2\src
filter_strategies3.calculate_port_in_sample_perf(
    data_path, 
    port_file_path, 
    logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path_train,
    port_perf_file_name=f'{file_name}_prob',
    # create_txt_code_port=filter_strategies3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over validation period
# the function will save the results inside portfolio_out_sample_performance in PonyGE2\src
portfolio_testing3.calculate_port_out_sample_perf(
    data_path, 
    port_file_path, 
    logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path,
    port_perf_file_name=f'{file_name}_prob',
    # create_txt_code_port=portfolio_testing3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over test period
# the function will save the results inside portfolio_test_sample_performance in PonyGE2\src
portfolio_testing3.calculate_port_test_sample_perf(
    data_path, 
    port_file_path, 
    logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path_test,
    port_perf_file_name=f'{file_name}_prob',
    # create_txt_code_port=portfolio_testing3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# creating portfolio weights by maximizing Sortino ratio over training period
# the function will save the results inside portfolio_strategies in PonyGE2\src
portfolio_testing3.creating_port_weights_mvp(
    lstr_path, 
    data_path=data_path, 
    lstr_file_name=lstr_file_name,
    port_path=port_path, 
    port_file_name=f"{file_name}",
    is_min_variance_port=False, # True when min_variance is used to derive weights
    is_sharpe_port=False, # True when Sharpe ratio is used to derive weights
    is_sortino_port=True, # True when Sortino ratio is used to derive weights
    prob_threshold=prob_threshold,
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq, 
    freq_minutes=time_freq,
    # create_txt_code=portfolio_testing3.create_txt_code_pnl1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over training period
# the function will save the results inside portfolio_in_sample_performance in PonyGE2\src
filter_strategies3.calculate_port_in_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_sortino.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path_train,
    port_perf_file_name='sortino',
    # create_txt_code_port=filter_strategies3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over validation period
# the function will save the results inside portfolio_out_sample_performance in PonyGE2\src
portfolio_testing3.calculate_port_out_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_sortino.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path,
    port_perf_file_name='sortino',
    # create_txt_code_port=portfolio_testing3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over test period
# the function will save the results inside portfolio_test_sample_performance in PonyGE2\src
portfolio_testing3.calculate_port_test_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_sortino.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path_test,
    port_perf_file_name='sortino',
    # create_txt_code_port=portfolio_testing3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# creating portfolio weights by maximizing Sharpe ratio over training period
# the function will save the results inside portfolio_strategies in PonyGE2\src
portfolio_testing3.creating_port_weights_mvp(
    lstr_path, 
    data_path=data_path, 
    port_path=port_path, 
    lstr_file_name=lstr_file_name,
    port_file_name=f"{file_name}",
    is_min_variance_port=False, # True when min_variance is used to derive weights
    is_sharpe_port=True, # True when Sharpe ratio is used to derive weights
    is_sortino_port=False, # True when Sortino ratio is used to derive weights
    prob_threshold=prob_threshold,
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq, 
    freq_minutes=time_freq,
    # create_txt_code=portfolio_testing3.create_txt_code_pnl1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over training period
# the function will save the results inside portfolio_in_sample_performance in PonyGE2\src
filter_strategies3.calculate_port_in_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_sharpe.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path_train,
    port_perf_file_name='sharpe',
    # create_txt_code_port=filter_strategies3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over validation period
# the function will save the results inside portfolio_out_sample_performance in PonyGE2\src
portfolio_testing3.calculate_port_out_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_sharpe.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path,
    port_perf_file_name='sharpe',
    # create_txt_code_port=portfolio_testing3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over test period
# the function will save the results inside portfolio_test_sample_performance in PonyGE2\src
portfolio_testing3.calculate_port_test_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_sharpe.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path_test,
    port_perf_file_name='sharpe',
    # create_txt_code_port=portfolio_testing3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# creating portfolio weights by minimizing variance/risk over training period
# the function will save the results inside portfolio_strategies in PonyGE2\src
portfolio_testing3.creating_port_weights_mvp(
    lstr_path, 
    data_path=data_path, 
    port_path=port_path,
    lstr_file_name=lstr_file_name, 
    port_file_name=f"{file_name}",
    is_min_variance_port=True, # True when min_variance is used to derive weights
    is_sharpe_port=False, # True when Sharpe ratio is used to derive weights
    is_sortino_port=False, # True when Sortino ratio is used to derive weights
    prob_threshold=prob_threshold,
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,  
    freq_minutes=time_freq,
    # create_txt_code=portfolio_testing3.create_txt_code_pnl1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over training period
# the function will save the results inside portfolio_in_sample_performance in PonyGE2\src
filter_strategies3.calculate_port_in_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_mvp.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path_train,
    port_perf_file_name='mvp',
    # create_txt_code_port=filter_strategies3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over validation period
# the function will save the results inside portfolio_out_sample_performance in PonyGE2\src
portfolio_testing3.calculate_port_out_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_mvp.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path,
    port_perf_file_name='mvp',
    # create_txt_code_port=portfolio_testing3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over test period
# the function will save the results inside portfolio_test_sample_performance in PonyGE2\src
portfolio_testing3.calculate_port_test_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_mvp.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path_test,
    port_perf_file_name='mvp',
    # create_txt_code_port=portfolio_testing3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# creating portfolio weights using hierarchical risk parity over training period
# the function will save the results inside portfolio_strategies in PonyGE2\src
portfolio_testing3.creating_port_weights_hrp(
    lstr_path, 
    data_path=data_path, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    port_path=port_path, 
    lstr_file_name=lstr_file_name,
    port_file_name=f"{file_name}_hrp",
    is_prob=True, 
    prob_threshold=prob_threshold,
    # create_txt_code=portfolio_testing3.create_txt_code_pnl1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over training period
# the function will save the results inside portfolio_in_sample_performance in PonyGE2\src
filter_strategies3.calculate_port_in_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_hrp.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path_train,
    port_perf_file_name='hrp',
    # create_txt_code_port=filter_strategies3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over validation period
# the function will save the results inside portfolio_out_sample_performance in PonyGE2\src
portfolio_testing3.calculate_port_out_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_hrp.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path,
    port_perf_file_name='hrp',
    # create_txt_code_port=portfolio_testing3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over test period
# the function will save the results inside portfolio_test_sample_performance in PonyGE2\src
portfolio_testing3.calculate_port_test_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_hrp.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path_test,
    port_perf_file_name='hrp',
    # create_txt_code_port=portfolio_testing3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# creating portfolio weights using kmeans over training period by selecting a strategy from each cluster 
# based on minimum distance from the cluster core
# the function will save the results inside portfolio_strategies in PonyGE2\src
portfolio_testing3.creating_port_weights_kmeans(
    lstr_path, 
    data_path=data_path, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    port_path=port_path, 
    lstr_file_name=lstr_file_name,
    port_file_name=f"{file_name}_kmeans_min_dist",
    is_prob=True, 
    prob_threshold=prob_threshold,
    select_method='min_dist',
    # create_txt_code=portfolio_testing3.create_txt_code_pnl1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over training period
# the function will save the results inside portfolio_in_sample_performance in PonyGE2\src
filter_strategies3.calculate_port_in_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_kmeans_min_dist.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path_train,
    port_perf_file_name='kmeans_min_dist',
    # create_txt_code_port=filter_strategies3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over validation period
# the function will save the results inside portfolio_out_sample_performance in PonyGE2\src
portfolio_testing3.calculate_port_out_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_kmeans_min_dist.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path,
    port_perf_file_name='kmeans_min_dist',
    # create_txt_code_port=portfolio_testing3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over test period
# the function will save the results inside portfolio_test_sample_performance in PonyGE2\src
portfolio_testing3.calculate_port_test_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_kmeans_min_dist.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path_test,
    port_perf_file_name='kmeans_min_dist',
    # create_txt_code_port=portfolio_testing3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# creating portfolio weights using kmeans over training period by selecting a strategy from each cluster 
# which has the minimum variance
# the function will save the results inside portfolio_strategies in PonyGE2\src
portfolio_testing3.creating_port_weights_kmeans(
    lstr_path, 
    data_path=data_path, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    port_path=port_path, 
    lstr_file_name=lstr_file_name,
    port_file_name=f"{file_name}_kmeans_min_var",
    is_prob=True, 
    prob_threshold=prob_threshold,
    select_method='min_var',
    # create_txt_code=portfolio_testing3.create_txt_code_pnl1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over training period
# the function will save the results inside portfolio_in_sample_performance in PonyGE2\src
filter_strategies3.calculate_port_in_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_kmeans_min_var.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path_train,
    port_perf_file_name='kmeans_min_var',
    # create_txt_code_port=filter_strategies3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over validation period
# the function will save the results inside portfolio_out_sample_performance in PonyGE2\src
portfolio_testing3.calculate_port_out_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_kmeans_min_var.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path,
    port_perf_file_name='kmeans_min_var',
    # create_txt_code_port=portfolio_testing3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over test period
# the function will save the results inside portfolio_test_sample_performance in PonyGE2\src
portfolio_testing3.calculate_port_test_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_kmeans_min_var.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path_test,
    port_perf_file_name='kmeans_min_var',
    # create_txt_code_port=portfolio_testing3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# creating portfolio weights using kmeans over training period by selecting a strategy from each cluster 
# which has maximum return
# the function will save the results inside portfolio_strategies in PonyGE2\src
portfolio_testing3.creating_port_weights_kmeans(
    lstr_path, 
    data_path=data_path, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    port_path=port_path, 
    lstr_file_name=lstr_file_name,
    port_file_name=f"{file_name}_kmeans_max_return",
    is_prob=True, 
    prob_threshold=prob_threshold,
    select_method='max_return',
    # create_txt_code=portfolio_testing3.create_txt_code_pnl1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over training period
# the function will save the results inside portfolio_in_sample_performance in PonyGE2\src
filter_strategies3.calculate_port_in_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_kmeans_max_return.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path_train,
    port_perf_file_name='kmeans_max_return',
    # create_txt_code_port=filter_strategies3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over validation period
# the function will save the results inside portfolio_out_sample_performance in PonyGE2\src
portfolio_testing3.calculate_port_out_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_kmeans_max_return.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path,
    port_perf_file_name='kmeans_max_return',
    # create_txt_code_port=portfolio_testing3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# calculating performance of the portfolio derived above over test period
# the function will save the results inside portfolio_test_sample_performance in PonyGE2\src
portfolio_testing3.calculate_port_test_sample_perf(
    data_path, 
    port_file_path=f"{port_path}/{file_name}_kmeans_max_return.csv", 
    logger=logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq,
    year=year,
    initial_amount=initial_amount,
    port_perf_path=port_perf_path_test,
    port_perf_file_name='kmeans_max_return',
    # create_txt_code_port=portfolio_testing3.create_txt_code_port1_vbt # use if you want to run the test using VectorBT
)

# calculating perfomance of all survived strategies over training period
# the function will create str_in_sample_performance folder inside PonyGE2\src to save the results
filter_strategies3.calculate_str_in_sample_perf(
    data_path, 
    port_file_path, 
    logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq, 
    year=year,
    initial_amount=initial_amount,
    str_perf_path=str_perf_path_train,
    str_perf_file_name=f'{file_name}_str_roi_train',
    # create_txt_code_port=filter_strategies3.create_txt_code1_vbt # use if you want to run the test using VectorBT
)

# calculating perfomance of all survived strategies over validation period
# the function will create str_out_sample_performance folder inside PonyGE2\src to save the results
portfolio_testing3.calculate_str_out_sample_perf(
    data_path, 
    port_file_path, 
    logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq, 
    year=year,
    initial_amount=initial_amount,
    str_perf_path=str_perf_path,
    str_perf_file_name=f'{file_name}_str_roi_valid',
    # create_txt_code_port=portfolio_testing3.create_txt_code1_vbt # use if you want to run the test using VectorBT
)

# calculating perfomance of all survived strategies over test period
# the function will create str_test_sample_performance folder inside PonyGE2\src to save the results
portfolio_testing3.calculate_str_test_sample_perf(
    data_path, 
    port_file_path, 
    logger, 
    n_fold=n_fold, 
    fold_size=fold_size, 
    time_freq=time_freq, 
    year=year,
    initial_amount=initial_amount,
    str_perf_path=str_perf_path_test,
    str_perf_file_name=f'{file_name}_str_roi_test',
    # create_txt_code_port=portfolio_testing3.create_txt_code1_vbt # use if you want to run the test using VectorBT
)