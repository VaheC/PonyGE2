from copy import copy
from os import getcwd, makedirs, path
from shutil import rmtree
import re

from algorithm.parameters import params
from utilities.stats import trackers


def save_stats_to_file(stats, end=False):
    """
    Write the results to a results file for later analysis

    :param stats: The stats.stats.stats dictionary.
    :param end: A boolean flag indicating whether or not the evolutionary
    process has finished.
    :return: Nothing.
    """

    if params['VERBOSE']:
        filename = path.join(params['FILE_PATH'], "stats.tsv")
        savefile = open(filename, 'a')
        for stat in sorted(stats.keys()):
            savefile.write(str(stats[stat]) + "\t")
        savefile.write("\n")
        savefile.close()

    elif end:
        filename = path.join(params['FILE_PATH'], "stats.tsv")
        savefile = open(filename, 'a')
        for item in trackers.stats_list:
            for stat in sorted(item.keys()):
                savefile.write(str(item[stat]) + "\t")
            savefile.write("\n")
        savefile.close()


def save_stats_headers(stats):
    """
    Saves the headers for all stats in the stats dictionary.

    :param stats: The stats.stats.stats dictionary.
    :return: Nothing.
    """

    filename = path.join(params['FILE_PATH'], "stats.tsv")
    savefile = open(filename, 'w')
    for stat in sorted(stats.keys()):
        savefile.write(str(stat) + "\t")
    savefile.write("\n")
    savefile.close()


def save_best_ind_to_file(stats, ind, end=False, name="best"):
    """
    Saves the best individual to a file.

    :param stats: The stats.stats.stats dictionary.
    :param ind: The individual to be saved to file.
    :param end: A boolean flag indicating whether or not the evolutionary
    process has finished.
    :param name: The name of the individual. Default set to "best".
    :return: Nothing.
    """

    filename = path.join(params['FILE_PATH'], (str(name) + ".txt"))
    savefile = open(filename, 'w')
    savefile.write("Generation:\n" + str(stats['gen']) + "\n\n")
    savefile.write("Phenotype:\n" + str(ind.phenotype) + "\n\n")
    savefile.write("Genotype:\n" + str(ind.genome) + "\n")
    savefile.write("Tree:\n" + str(ind.tree) + "\n")
    if hasattr(params['FITNESS_FUNCTION'], "training_test"):
        if end:
            savefile.write("\nTraining fitness:\n" + str(ind.training_fitness))
            savefile.write("\nTest fitness:\n" + str(ind.test_fitness))
        else:
            savefile.write("\nFitness:\n" + str(ind.fitness))
    else:
        savefile.write("\nFitness:\n" + str(ind.fitness))
    savefile.close()


def save_first_front_to_file(stats, end=False, name="first"):
    """
    Saves all individuals in the first front to individual files in a folder.

    :param stats: The stats.stats.stats dictionary.
    :param end: A boolean flag indicating whether or not the evolutionary
                process has finished.
    :param name: The name of the front folder. Default set to "first_front".
    :return: Nothing.
    """

    # Save the file path (we will be over-writing it).
    orig_file_path = copy(params['FILE_PATH'])

    # Define the new file path.
    params['FILE_PATH'] = path.join(orig_file_path, str(name) + "_front")

    # Check if the front folder exists already
    if path.exists(params['FILE_PATH']):
        # Remove previous files.
        rmtree(params['FILE_PATH'])

    # Create front folder.
    makedirs(params['FILE_PATH'])

    for i, ind in enumerate(trackers.best_ever):
        # Save each individual in the first front to file.
        save_best_ind_to_file(stats, ind, end, name=str(i))

    # Re-set the file path.
    params['FILE_PATH'] = copy(orig_file_path)


def generate_folders_and_files():
    """
    Generates necessary folders and files for saving statistics and parameters.

    :return: Nothing.
    """

    if params['EXPERIMENT_NAME']:
        # Experiment manager is being used.
        path_1 = path.join(getcwd(), "..", "results")

        if not path.isdir(path_1):
            # Create results folder.
            makedirs(path_1, exist_ok=True)

        # Set file path to include experiment name.
        params['FILE_PATH'] = path.join(path_1, params['EXPERIMENT_NAME'])

    else:
        # Set file path to results folder.
        params['FILE_PATH'] = path.join(getcwd(), "..", "results")

    # Generate save folders
    if not path.isdir(params['FILE_PATH']):
        makedirs(params['FILE_PATH'], exist_ok=True)

    if not path.isdir(path.join(params['FILE_PATH'],
                                str(params['TIME_STAMP']))):
        makedirs(path.join(params['FILE_PATH'],
                           str(params['TIME_STAMP'])), exist_ok=True)

    params['FILE_PATH'] = path.join(params['FILE_PATH'],
                                    str(params['TIME_STAMP']))

    save_params_to_file()


def save_params_to_file():
    """
    Save evolutionary parameters in a parameters.txt file.

    :return: Nothing.
    """

    # Generate file path and name.
    filename = path.join(params['FILE_PATH'], "parameters.txt")
    savefile = open(filename, 'w')

    # Justify whitespaces for pretty printing/saving.
    col_width = max(len(param) for param in params.keys())

    for param in sorted(params.keys()):
        # Create whitespace buffer for pretty printing/saving.
        spaces = [" " for _ in range(col_width - len(param))]
        savefile.write(str(param) + ": " + "".join(spaces) +
                       str(params[param]) + "\n")

    savefile.close()

def save_all_ind_to_file(inds, end=False, name="ge_results"):
    """
    Saves the all individuals to a file.

    :param inds: The individuals to be saved to file.
    :param end: A boolean flag indicating whether or not the evolutionary
    process has finished.
    :param name: The name of the file to be saved
    :return: Nothing.
    """

    filename = path.join(params['FILE_PATH'], (str(name) + ".csv"))
    savefile = open(filename, 'w')

    # savefile.write(f"buy;sell;fitness\n\n")
    # savefile.write(f"buy;exit_buy;sell;exit_sell;fitness\n")
    savefile.write(f"alpha;fitness\n")

    i = 0
    for k, v in inds.items():

        # buy_signal = re.findall(r"trading_signals\(buy_signal\=(.*)\, sell_signal", k)[0]
        # sell_signal = re.findall(r"\, sell_signal\=(.*)\)", k)[0]

        # savefile.write(f"{buy_signal};{sell_signal};{v}\n\n")

        # buy_signal = re.findall(r"trading_signals_buy\(buy_signal\=(.*)\, exit_signal", k)[0]
        # buy_exit_signal = re.findall(r"\, exit_signal\=(.*)\)", k)[0]
        # sell_signal = re.findall(r"trading_signals_sell\(sell_signal\=(.*)\, exit_signal", k)[0]
        # sell_exit_signal = re.findall(r"\, exit_signal\=(.*)\)", k)[0]

        # savefile.write(f"{buy_signal};{buy_exit_signal};{sell_signal};{sell_exit_signal};{v}\n")
        if i == 0:
            print(k)
        i += 1
        alpha = re.findall(r"alpha_arr \= (.*)", k)[0]
        savefile.write(f"{alpha};{v}\n")

    savefile.close()