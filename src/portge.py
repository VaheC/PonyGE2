from pathlib import Path

from datetime import datetime

from fitness.performance import filter_strategies, portfolio_testing
from fitness.custom_logger.load_logger import create_logger

# set the path to the training data, used to get strategies, below
data_path = Path(r'C:/\Users/\vchar/\OneDrive/\Desktop/\ML Projects/\Upwork/\AlgoT_ML_Dev/\GrammarEvolution/\PonyGE2/\all_data_1min_all.csv')

# set the path to the csv file containing the strategies generated
# strategy_file_path = Path(r"C:/\Users/\vchar/\Downloads/\ge_results (5).csv")
strategy_file_path = Path(r"C:/\Users/\vchar/\Downloads/\ge_results (6).csv")

stats_path = 'testing_results' # folder which will contain csv files with testing stats as in testing_strategies.ipynb
file_name = 'fold2_baseline' # suffix which will be added to the csv files created during this run
str_path = 'selected_strategies' # folder which will contain csv files with strategies survived after testing
selected_str_file_path = f'{str_path}/selected_strategies_{file_name}.csv' # file path which will contain the survived strategies
lstr_path = 'live_strategies' # folder which will contain csv files with strategies survived after out of fold/sample testing
lstr_file_name='fold2_baseline' # suffix which will be added to the csv files containing the strategies survived after out of fold/sample teting
port_path = 'portfolio_strategies' # folder which will contain csv files with the portfolio weights of the survived strategies
port_file_path = f'{port_path}/portfolio_{file_name}.csv' # path to a file containing the portfolio weigths of the survived strategies
port_perf_path = 'portfolio_out_sample_performance' # folder which will contain csv files with the performance (ROI (%), PNL) of the portfolio for different out of fold/sample data

str_date = datetime.now().strftime('%Y%m%d_%H:%M')
    
logger = create_logger(filename=f'log_{str_date}.log')

filter_strategies.save_stats(
    data_path, strategy_file_path, n_fold=2, logger=logger,
    stats_path=stats_path, stats_file_name=file_name, 
    n_bars=50400
)

filter_strategies.filter_save_strategies(strategy_file_path, logger, 
    stats_path=stats_path, stats_file_name=file_name, str_path=str_path, 
    str_file_name=file_name, entry_testing_threshold=50,
    exit_testing_threshold=50, core_testing_threshold=60,
    prob_threshold=0.98
)

portfolio_testing.filter_save_lstr(data_path, n_fold=2, str_file_path=selected_str_file_path, logger=logger,
    lstr_path=lstr_path, lstr_file_name=lstr_file_name, is_subset=False, start_subset=0, end_subset=10,
    entry_testing_threshold=50, exit_testing_threshold=50, core_testing_threshold=60,
    prob_threshold=0.8
)

portfolio_testing.creating_port_weights(lstr_path, logger, port_path=port_path, 
    port_file_name=file_name, is_prob=False, prob_threshold=0.9
)

portfolio_testing.calculate_port_out_sample_perf(
    data_path, port_file_path, logger, n_bars=50400, 
    initial_amount=350000, start_fold=3, end_fold=9, 
    port_perf_path=port_perf_path,
    port_perf_file_name=file_name
)