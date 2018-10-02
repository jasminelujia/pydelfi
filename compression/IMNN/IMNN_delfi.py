import sys
sys.path.insert(0, 'compression/IMNN')

import numpy as np
import IMNN
import tensorflow as tf
import pickle
import tqdm

def train_IMNN(simulator, simulator_args, theta, der, initial_sims, filename, num_epochs, proposal, proposal_sampler, proposal_sims, make_simulations = True):

    tf.reset_default_graph()

    partial_fraction = 0.05
    if make_simulations:
        data = []
        data_m = []
        data_p = []
        partial_sims = int(initial_sims * partial_fraction)
        print("Partial sims = ", partial_sims)
        for sim in tqdm.trange(initial_sims * 2):
            data.append(simulator(theta, simulator_args).flatten())
            if sim < 2 * partial_sims:
                seed = np.random.randint(1e6)
                data_m.append([])
                data_p.append([])
                for param in range(len(theta)):
                    np.random.seed(seed)
                    data_m[-1].append(simulator(der[0], simulator_args).flatten())
                    np.random.seed(seed)
                    data_p[-1].append(simulator(der[1], simulator_args).flatten())
        data_check = []
        for sim in tqdm.trange(proposal_sims):
            theta = proposal_sampler.draw()
            data_check.append(simulator(theta, simulator_args).flatten())

        data = {'x_central': np.array(data)[:len(data)//2],
                'x_m': np.array(data_m)[:len(data_m)//2],
                'x_p': np.array(data_p)[:len(data_p)//2],
                'x_central_test': np.array(data)[len(data)//2:],
                'x_m_test': np.array(data_m)[len(data_m)//2:],
                'x_p_test': np.array(data_p)[len(data_p)//2:],
                'x_check': np.array(data_check),
                }

        print(data['x_central'].shape, data['x_m'].shape, data['x_p'].shape, data['x_central_test'].shape, data['x_m_test'].shape, data['x_p_test'].shape, data['x_check'].shape)

        with open("simulations/" + filename + ".pickle", 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:

        with open("simulations/" + filename + ".pickle", 'rb') as handle:
            data = pickle.load(handle)

        print(data['x_central'].shape, data['x_m'].shape, data['x_p'].shape, data['x_central_test'].shape, data['x_m_test'].shape, data['x_p_test'].shape, data['x_check'].shape)

    der_den = 1. / (der[1] - der[0])

    hidden_layers = []
    nodes = data['x_central'].shape[-1]
    while nodes > 5 * len(theta):
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
        'covariance regularisation': True,
        'number of simulations for covariance regularisation': proposal_sims,
        'comparison covariance': proposal,
        'prebuild': True,
        'input shape': list(data['x_central'].shape[1:]),
        'preload data': data,
        'save file': "compression/cosmic_shear",
        'wv': 0.,
        'bb': 0.1,
        'activation': tf.nn.leaky_relu,
        'α': 0.01,
        'hidden layers': hidden_layers
        }

    n = IMNN.IMNN(parameters = parameters)
    η = 1e-2
    n.setup(η = η)

    train_F, test_F = n.train(num_epochs = num_epochs, n_train = 1, keep_rate = 0.8)

    return train_F, test_F, n

def IMNN_compressor(data, n):
    data = data.flatten()
    data = data.reshape((1, data.shape[0]))
    return n.sess.run(n.MLE, feed_dict = {n.x: data, n.dropout: 1.})[0]
