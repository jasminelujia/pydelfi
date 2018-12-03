import sys
sys.path.insert(0, 'compression/IMNN/IMNN')

import numpy as np
import IMNN
import tensorflow as tf
from multiprocessing import Pool
from functools import partial
import os
import pickle
import tqdm

def make_simulations_without_derivatives(sim_number, data_arrays):
    theta = data_arrays[0]
    simulator = data_arrays[1]
    simulator_args = data_arrays[2]
    data = simulator(theta, np.random.randint(1e8), simulator_args)
    data_test = simulator(theta, np.random.randint(1e8), simulator_args)
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
    np.random.seed()
    seed_1 = np.random.randint(1e8)
    seed_2 = np.random.randint(1e8)
    for param in range(len(theta)):
        theta_m = np.copy(theta)
        theta_m[param] -= der[param]
        data_m[param] = simulator(theta_m, seed_1, simulator_args)
        data_m_test[param] = simulator(theta_m, seed_2, simulator_args)
        theta_p = np.copy(theta)
        theta_p[param] += der[param]
        data_p[param] = simulator(theta_p, seed_1, simulator_args)
        data_p_test[param] = simulator(theta_p, seed_2, simulator_args)
    return data_m, data_m_test, data_p, data_p_test

def make_simulations_with_known_derivatives(sim_number, data_arrays):
    theta = data_arrays[0]
    simulator = data_arrays[1]
    simulator_args = data_arrays[2]
    partial_sims = data_arrays[3]
    derivative_simulator = data_arrays[4]
    h = data_arrays[5]
    derivative_args = data_arrays[6]
    seed_1 = np.random.randint(1e8)
    seed_2 = np.random.randint(1e8)
    data = simulator(theta, seed_1, simulator_args)
    data_test = simulator(theta, seed_2, simulator_args)
    if sim_number < partial_sims:
        data_d = derivative_simulator(theta, seed_1, h, derivative_args)
        data_d_test = derivative_simulator(theta, seed_2, h, derivative_args)
        return data, data_d, data_test, data_d_test
    return data, data_test

def get_network(simulator, simulator_args, simulation_shape, theta, der, initial_sims, partial_fraction, filename, η, make_simulations = True, partials_only = False, true_derivatives = False, derivative_simulator = None, derivative_args = None, load_network = False):

    tf.reset_default_graph()

    partial_sims = int(partial_fraction * initial_sims)
    print("number of partial simulations = ", partial_sims)
    if make_simulations:
        data = np.zeros([initial_sims] + simulation_shape)
        data_test = np.zeros([initial_sims] + simulation_shape)
        if true_derivatives:
            data_d = np.zeros([int(initial_sims * partial_fraction), len(theta)] + simulation_shape)
            data_d_test = np.zeros([int(initial_sims * partial_fraction), len(theta)] + simulation_shape)
        else:
            data_m = np.zeros([int(initial_sims * partial_fraction), len(theta)] + simulation_shape)
            data_p = np.zeros([int(initial_sims * partial_fraction), len(theta)] + simulation_shape)
            data_m_test = np.zeros([int(initial_sims * partial_fraction), len(theta)] + simulation_shape)
            data_p_test = np.zeros([int(initial_sims * partial_fraction), len(theta)] + simulation_shape)


        pool = Pool(os.cpu_count())

        if true_derivatives:
            counter = 0
            der_counter = 0
            for i in tqdm.tqdm(pool.imap_unordered(partial(make_simulations_with_known_derivatives, data_arrays = [theta, simulator, simulator_args, partial_sims, derivative_simulator, der, derivative_args]), np.arange(initial_sims)), desc = "Fiducial simulations", total = initial_sims):
                if len(i) == 4:
                    if counter > der_counter:
                        data[counter] = data[der_counter]
                        data_test[counter] = data_test[der_counter]
                    data[der_counter] = i[0]
                    data_d[der_counter] = i[1]
                    data_test[der_counter] = i[2]
                    data_d_test[der_counter] = i[3]
                    der_counter += 1
                else:
                    data[counter] = i[0]
                    data_test[counter] = i[1]
                counter += 1
        else:
            if not partials_only:
                counter = 0
                for i in tqdm.tqdm(pool.imap_unordered(partial(make_simulations_without_derivatives, data_arrays = [theta, simulator, simulator_args]), np.arange(initial_sims)), desc = "Fiducial simulations", total = initial_sims):
                    data[counter] = i[0]
                    data_test[counter] = i[1]
                    counter += 1
            else:
                with open("simulations/" + filename + ".pickle", 'rb') as handle:
                    data = pickle.load(handle)
            counter = 0
            for i in tqdm.tqdm(pool.imap_unordered(partial(make_simulations_with_derivatives, data_arrays = [theta, der, simulator, simulator_args, simulation_shape]), np.arange(int(initial_sims * partial_fraction))), desc = "Derivative simulations", total = int(initial_sims * partial_fraction)):
                data_m[counter] = i[0]
                data_m_test[counter] = i[1]
                data_p[counter] = i[2]
                data_p_test[counter] = i[3]
                counter += 1
            pool.close()

        if true_derivatives:
            data = {'x_central': np.array(data)[:initial_sims],
                    'x_d': np.array(data_d)[:partial_sims],
                    'x_central_test': np.array(data_test)[:initial_sims],
                    'x_d_test': np.array(data_d_test)[:partial_sims],
                    }
        else:
            if not partials_only:
                data = {'x_central': np.array(data)[:initial_sims],
                        'x_m': np.array(data_m)[:partial_sims],
                        'x_p': np.array(data_p)[:partial_sims],
                        'x_central_test': np.array(data_test)[:initial_sims],
                        'x_m_test': np.array(data_m_test)[:partial_sims],
                        'x_p_test': np.array(data_p_test)[:partial_sims],
                        }
            else:
                data['x_m'] = np.array(data_m)[:partial_sims]
                data['x_p'] = np.array(data_p)[:partial_sims]
                data['x_m_test'] = np.array(data_m_test)[:partial_sims]
                data['x_p_test'] = np.array(data_p_test)[:partial_sims]

        with open("simulations/" + filename + ".pickle", 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:

        with open("simulations/" + filename + ".pickle", 'rb') as handle:
            data = pickle.load(handle)

    hidden_layers = []
    nodes = int(np.prod(simulation_shape))
    while nodes > len(theta):
        hidden_layers.append(nodes)
        nodes = nodes // 2
    print(hidden_layers)

    parameters = {
        'verbose': False,
        'number of simulations': initial_sims,
        'fiducial θ': theta,
        'differentiation fraction': partial_fraction,
        'number of summaries': len(theta),
        'calculate MLE': True,
        'prebuild': True,
        'input shape': list(data['x_central'].shape[1:]),
        'save file': "compression/cosmic_shear",
        'wv': 0.,
        'bb': 0.1,
        'activation': tf.nn.leaky_relu,
        'α': 0.1,
        'hidden layers': hidden_layers
        }

    if 'x_m' in data.keys():
        data['x_central'] = data['x_central'][:initial_sims]
        data['x_m'] = data['x_m'][:partial_sims]
        data['x_p'] = data['x_p'][:partial_sims]
        data['x_central_test'] = data['x_central_test'][:initial_sims]
        data['x_m_test'] = data['x_m_test'][:partial_sims]
        data['x_p_test'] = data['x_p_test'][:partial_sims]
        der_den = 1. / ((theta + der) - (theta - der))
        parameters['true derivative'] = False
        parameters['derivative denominator'] = der_den
    else:
        data['x_central'] = data['x_central'][:initial_sims]
        data['x_d'] = data['x_d'][:partial_sims]
        data['x_central_test'] = data['x_central_test'][:initial_sims]
        data['x_d_test'] = data['x_d_test'][:partial_sims]
        parameters['true derivative'] = True

    parameters['preload data'] = data

    n = IMNN.IMNN(parameters = parameters)
    if load_network:
        n.restore_network()
    else:
        n.setup(η = η)
    return n

def train_IMNN(n, num_epochs, to_continue = False):
    n.train(num_epochs = num_epochs, n_train = 1, keep_rate = 1., to_continue = to_continue)

def IMNN_compressor(data, n):
    data = data.reshape([1] + list(data.shape))
    return n.sess.run(n.output, feed_dict = {n.x: data, n.dropout: 1.})[0]

def IMNN_MLE(data, n):
    data = data.reshape([1] + list(data.shape))
    return n.sess.run(n.MLE, feed_dict = {n.x: data, n.dropout: 1.})[0]

def plot_train_history(compression_network):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize = (10, 8))
    plt.subplots_adjust(wspace = 0.2)
    ax.set_yscale("log")
    ax.plot(np.arange(1, len(compression_network.history["det(F)"]) + 1), compression_network.history["det(F)"], color = "C0", label = r"Training $|{\bf F}_{\alpha\beta}|$")
    ax.plot(np.arange(1, len(compression_network.history["det(test F)"]) + 1), compression_network.history["det(test F)"], color = "C1", label = r"Validation $|{\bf F}_{\alpha\beta}|$")
    ax.legend(fontsize = 20)
    ax.set_ylabel(r"$|{\bf F}_{\alpha\beta}|$", fontsize = 20)
    ax.set_xlim([1, len(compression_network.history["det(F)"])])
    #ax[1].plot(np.arange(1, len(compression_network.history["Λ"]) + 1), compression_network.history["Λ"], color = "C0")
    #ax[1].plot(np.arange(1, len(compression_network.history["test Λ"]) + 1), compression_network.history["test Λ"], color = "C1")
    #ax[1].set_ylabel(r"$\Lambda$", fontsize = 20)
    #ax[1].set_xlabel("Number of epochs", fontsize = 20)
    ax.set_xlabel("Number of epochs", fontsize = 20)
    plt.savefig("Training_curve.pdf", bbox_inches = "tight")
    #ax[1].set_xlim([1, len(compression_network.history["det(F)"])]);
