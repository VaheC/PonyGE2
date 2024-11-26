from multiprocessing import Pool

from algorithm.parameters import params
from fitness.evaluation import evaluate_fitness
from operators.initialisation import initialisation
from stats.stats import get_stats, stats
from utilities.algorithm.initialise_run import pool_init
from utilities.stats import trackers

# import tensorflow as tf

# def use_tpu():
#     # Check if TPU is available and connect
#     try:
#         # Initialize TPU
#         tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  
#         print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
        
#         # Connect to TPU and initialize the system
#         tf.config.experimental_connect_to_cluster(tpu)
#         tf.tpu.experimental.initialize_tpu_system(tpu)
        
#         # Create a distribution strategy for TPU
#         strategy = tf.distribute.TPUStrategy(tpu)
#     except ValueError:
#         print('No TPU found. Falling back to CPU or GPU.')
#         strategy = tf.distribute.get_strategy()  # Use CPU or GPU fallback

#     # Use the TPU strategy scope to execute code on the TPU
#     with strategy.scope():
#         # Create two random tensors
#         tensor_a = tf.random.uniform(shape=[3, 3], minval=0, maxval=10, dtype=tf.float32)
#         tensor_b = tf.random.uniform(shape=[3, 3], minval=0, maxval=10, dtype=tf.float32)
        
#         # Add the two tensors
#         result = tf.add(tensor_a, tensor_b)

def search_loop():
    """
    This is a standard search process for an evolutionary algorithm. Loop over
    a given number of generations.
    
    :return: The final population after the evolutionary process has run for
    the specified number of generations.
    """

    if params['MULTICORE']:
        # initialize pool once, if multi-core is enabled
        params['POOL'] = Pool(processes=params['CORES'], initializer=pool_init,
                              initargs=(params,))  # , maxtasksperchild=1)

    # Initialise population
    individuals = initialisation(params['POPULATION_SIZE'])

    # Evaluate initial population
    individuals = evaluate_fitness(individuals)

    # Generate statistics for run so far
    get_stats(individuals)

    # Traditional GE
    for generation in range(1, (params['GENERATIONS'] + 1)):
        stats['gen'] = generation
        # use_tpu()
        # New generation
        individuals = params['STEP'](individuals)

    if params['MULTICORE']:
        # Close the workers pool (otherwise they'll live on forever).
        params['POOL'].close()

    return individuals

def search_loop_from_state():
    """
    Run the evolutionary search process from a loaded state. Pick up where
    it left off previously.

    :return: The final population after the evolutionary process has run for
    the specified number of generations.
    """

    individuals = trackers.state_individuals

    if params['MULTICORE']:
        # initialize pool once, if multi-core is enabled
        params['POOL'] = Pool(processes=params['CORES'], initializer=pool_init,
                              initargs=(params,))  # , maxtasksperchild=1)

    # Traditional GE
    for generation in range(stats['gen'] + 1, (params['GENERATIONS'] + 1)):
        stats['gen'] = generation

        # New generation
        individuals = params['STEP'](individuals)

    if params['MULTICORE']:
        # Close the workers pool (otherwise they'll live on forever).
        params['POOL'].close()

    return individuals
