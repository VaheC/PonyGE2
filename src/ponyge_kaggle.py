#! /usr/bin/env python

# PonyGE2
# Copyright (c) 2017 Michael Fenton, James McDermott,
#                    David Fagan, Stefan Forstenlechner,
#                    and Erik Hemberg
# Hereby licensed under the GNU GPL v3.
""" Python GE implementation """

from utilities.algorithm.general import check_python_version

check_python_version()

from stats.stats import get_stats
from algorithm.parameters import params, set_params
import sys

import tensorflow as tf

def use_tpu():
    # Check if TPU is available and connect
    try:
        # Initialize TPU
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
        
        # Connect to TPU and initialize the system
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        
        # Create a distribution strategy for TPU
        strategy = tf.distribute.TPUStrategy(tpu)
    except ValueError:
        print('No TPU found. Falling back to CPU or GPU.')
        strategy = tf.distribute.get_strategy()  # Use CPU or GPU fallback

    # Use the TPU strategy scope to execute code on the TPU
    with strategy.scope():
        # Create two random tensors
        tensor_a = tf.random.uniform(shape=[3, 3], minval=0, maxval=10, dtype=tf.float32)
        tensor_b = tf.random.uniform(shape=[3, 3], minval=0, maxval=10, dtype=tf.float32)
        
        # Add the two tensors
        result = tf.add(tensor_a, tensor_b)


def mane():
    """ Run program """
    set_params(sys.argv[1:])  # exclude the ponyge.py arg itself

    # Run evolution
    individuals = params['SEARCH_LOOP']()

    # Print final review
    get_stats(individuals, end=True)


if __name__ == "__main__":
    mane()
    use_tpu()
