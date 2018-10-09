import sys
sys.path.insert(0, 'compression/IMNN')

import numpy as np
import IMNN
import tensorflow as tf
from multiprocessing import Pool
from functools import partial
import os
import pickle
import tqdm

def make_simulations_without_derivatives(sim_number, data_arrays):
    #print(sim_number)
    theta = data_arrays[0]
    simulator = data_arrays[1]
    simulator_args = data_arrays[2]
    data = simulator(theta, simulator_args).flatten()
    data_test = simulator(theta, simulator_args).flatten()
    return data, data_test

def make_simulations_with_derivatives(sim_number, data_arrays):
    theta = data_arrays[0]
    der = data_arrays[1]
    simulator = data_arrays[2]
    simulator_args = data_arrays[3]
    shape = data_arrays[4]
    data_m = np.zeros([len(theta)] + shape)
    data_m_test = np.zeros([len(theta)] + shape)
    data_p = np.zeros([len(theta)] + shape)
    data_p_test = np.zeros([len(theta)] + shape)
    for param in range(len(theta)):
        seed = np.random.randint(1e6)
        theta_m = np.copy(theta)
        theta_m[param] -= der[param]
        np.random.seed(seed)
        data_m[param] = simulator(theta_m, simulator_args).flatten()
        data_m_test[param] = simulator(theta_m, simulator_args).flatten()
        theta_p = np.copy(theta)
        theta_p[param] += der[param]
        np.random.seed(seed)
        data_p[param] = simulator(theta_p, simulator_args).flatten()
        data_p_test[param] = simulator(theta_p, simulator_args).flatten()
        print(theta_m, theta, theta_p)
    return data_m, data_m_test, data_p, data_p_test

def get_network(simulator, simulator_args, theta, der, initial_sims, filename, make_simulations = True):

    tf.reset_default_graph()

    partial_fraction = 0.1
    if make_simulations:
        first = simulator(theta, simulator_args).flatten()
        data = np.zeros([initial_sims] + list(first.shape))
        data_m = np.zeros([int(initial_sims * partial_fraction), len(theta)] + list(first.shape))
        data_p = np.zeros([int(initial_sims * partial_fraction), len(theta)] + list(first.shape))
        data_test = np.zeros([initial_sims] + list(first.shape))
        data_m_test = np.zeros([int(initial_sims * partial_fraction), len(theta)] + list(first.shape))
        data_p_test = np.zeros([int(initial_sims * partial_fraction), len(theta)] + list(first.shape))
        pool = Pool(os.cpu_count())
        counter = 0
        for i in tqdm.tqdm(pool.imap_unordered(partial(make_simulations_with_derivatives, data_arrays = [theta, der, simulator, simulator_args, list(first.shape)]), np.arange(int(initial_sims * partial_fraction))), desc = "Derivative simulations", total = int(initial_sims * partial_fraction)):
            data_m[counter] = i[0]
            data_m_test[counter] = i[1]
            data_p[counter] = i[2]
            data_p_test[counter] = i[3]
            counter += 1
        counter = 0
        for i in tqdm.tqdm(pool.imap_unordered(partial(make_simulations_without_derivatives, data_arrays = [theta, simulator, simulator_args]), np.arange(int(initial_sims * partial_fraction), initial_sims)), desc = "Fiducial simulations", total = initial_sims):
            data[counter] = i[0]
            data_test[counter] = i[1]
            counter += 1

        pool.close()

        data = {'x_central': np.array(data),
                'x_m': np.array(data_m),
                'x_p': np.array(data_p),
                'x_central_test': np.array(data_test),
                'x_m_test': np.array(data_m_test),
                'x_p_test': np.array(data_p_test),
                }

        print(data['x_central'].shape, data['x_m'].shape, data['x_p'].shape, data['x_central_test'].shape, data['x_m_test'].shape, data['x_p_test'].shape)

        with open("simulations/" + filename + ".pickle", 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:

        with open("simulations/" + filename + ".pickle", 'rb') as handle:
            data = pickle.load(handle)

        print(data['x_central'].shape, data['x_m'].shape, data['x_p'].shape, data['x_central_test'].shape, data['x_m_test'].shape, data['x_p_test'].shape)

    der_den = 1. / (2 * der)

    hidden_layers = []
    nodes = data['x_central'].shape[-1]
    while nodes > len(theta):
        hidden_layers.append(nodes)
        nodes = nodes // 2

    print(hidden_layers)
    parameters = {
        'verbose': False,
        'number of simulations': initial_sims,
        'fiducial θ': theta,
        'derivative denominator': der_den,
        'differentiation fraction': partial_fraction,
        'number of summaries': len(theta),
        'calculate MLE': True,
        'prebuild': True,
        'input shape': list(data['x_central'].shape[1:]),
        'preload data': data,
        'save file': "compression/cosmic_shear",
        'wv': 0.,
        'bb': 0.1,
        'activation': tf.nn.leaky_relu,
        'α': 0.1,
        'hidden layers': hidden_layers
        }

    n = IMNN.IMNN(parameters = parameters)
    η = 1e-1
    n.setup(η = η)
    return n

def train_IMNN(n, num_epochs):
    n.train(num_epochs = num_epochs, n_train = 1, keep_rate = 0.8)

def IMNN_compressor(data, n):
    data = data.flatten()
    data = data.reshape((1, data.shape[0]))
    return n.sess.run(n.MLE, feed_dict = {n.x: data, n.dropout: 1.})[0]
