import numpy as np
import numpy.random as rng
import tensorflow as tf
import ndes.mades

dtype = tf.float32

class ConditionalMaskedAutoregressiveFlow:
    """
    Implements a Conditional Masked Autoregressive Flow.
    """

    def __init__(self, n_parameters, n_data, n_hiddens, act_fun, n_mades,
                 output_order='sequential', mode='sequential', input_parameters=None, input_data=None, logpdf=None, index=1):
        """
        Constructor.
        :param n_parameters: number of (conditional) inputs
        :param n_data: number of outputs
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param act_fun: tensorflow activation function
        :param n_mades: number of mades in the flow
        :param output_order: order of outputs of last made
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        :param input: tensorflow placeholder to serve as input; if None, a new placeholder is created
        :param output: tensorflow placeholder to serve as output; if None, a new placeholder is created
        """

        # save input arguments
        self.n_parameters = n_parameters
        self.n_data = n_data
        self.n_hiddens = n_hiddens
        self.act_fun = act_fun
        self.n_mades = n_mades
        self.mode = mode

        self.parameters = tf.placeholder(dtype=dtype,shape=[None,n_parameters],name='parameters') if input_parameters is None else input_parameters
        self.data = tf.placeholder(dtype=dtype,shape=[None,n_data],name='data') if input_data is None else input_data
        self.logpdf = tf.placeholder(dtype=dtype,shape=[None],name='logpdf') if logpdf is None else logpdf
        self.parms = []

        self.mades = []
        self.bns = []
        self.u = self.data
        self.logdet_dudy = 0.0

        for i in range(n_mades):
            
            # create a new made
            with tf.variable_scope('nde_' + str(index) + '_made_' + str(i + 1)):
                made = ndes.mades.ConditionalGaussianMade(n_parameters, n_data, n_hiddens, act_fun,
                                                 output_order, mode, self.parameters, self.u)
            self.mades.append(made)
            self.parms += made.parms
            output_order = output_order if output_order is 'random' else made.output_order[::-1]

            # inverse autoregressive transform
            self.u = made.u
            self.logdet_dudy += 0.5 * tf.reduce_sum(made.logp, axis=1,keepdims=True)

        self.output_order = self.mades[0].output_order

        # log likelihoods
        self.L = tf.add(-0.5 * n_data * np.log(2 * np.pi) - 0.5 * tf.reduce_sum(self.u ** 2, axis=1,keepdims=True), self.logdet_dudy,name='L')

        # train objective
        self.trn_loss = -tf.reduce_mean(self.L,name='trn_loss')

    def eval(self, xy, sess, log=True):
        """
        Evaluate log probabilities for given input-output pairs.
        :param xy: a pair (x, y) where x rows are inputs and y rows are outputs
        :param sess: tensorflow session where the graph is run
        :param log: whether to return probabilities in the log domain
        :return: log probabilities: log p(y|x)
        """
        
        x, y = xy
        lprob = sess.run(self.L,feed_dict={self.parameters:x,self.data:y})[0]

        return lprob if log else np.exp(lprob)

    def gen(self, x, sess, n_samples=1, u=None):
        """
        Generate samples, by propagating random numbers through each made, after conditioning on input x.
        :param x: input vector
        :param sess: tensorflow session where the graph is run
        :param n_samples: number of samples
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        y = rng.randn(n_samples, self.n_data).astype(dtype) if u is None else u

        if getattr(self, 'batch_norm', False):

            for made, bn in zip(self.mades[::-1], self.bns[::-1]):
                y = bn.eval_inv(sess,y)
                y = made.gen(x, sess, n_samples, y)

        else:

            for made in self.mades[::-1]:
                y = made.gen(x, sess, n_samples, y)

        return y

    def calc_random_numbers(self, xy):
        """
        Givan a dataset, calculate the random numbers used internally to generate the dataset.
        :param xy: a pair (x, y) of numpy arrays, where x rows are inputs and y rows are outputs
        :return: numpy array, rows are corresponding random numbers
        """

        x, y = xy
        return sess.run(self.u,feed_dict={self.parameters:x,self.data:y})


class MixtureDensityNetwork:
    """
    Implements a Mixture Density Network for modeling p(y|x)
    """

    def __init__(self, n_parameters, n_data, n_components = 3, n_hidden=[50,50], activations=[tf.tanh, tf.tanh],
                 input_parameters=None, input_data=None, logpdf=None, index=1):
        """
        Constructor.
        :param n_parameters: number of (conditional) inputs
        :param n_data: number of outputs (ie dimensionality of distribution you're parameterizing)
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param activations: tensorflow activation functions for each hidden layer
        :param input: tensorflow placeholder to serve as input; if None, a new placeholder is created
        :param output: tensorflow placeholder to serve as output; if None, a new placeholder is created
        """
        
        # save input arguments
        self.n_parameters = n_parameters
        self.n_data = n_data
        self.M = n_components
        self.N = int((self.n_data + self.n_data * (self.n_data + 1) / 2 + 1)*self.M)
        self.n_hidden = n_hidden
        self.activations = activations
        
        self.parameters = tf.placeholder(dtype=dtype,shape=[None,self.n_parameters],name='parameters') if input_parameters is None else input_parameters
        self.data = tf.placeholder(dtype=dtype,shape=[None,self.n_data],name='data') if input_data is None else input_data
        self.logpdf = tf.placeholder(dtype=dtype,shape=[None],name='logpdf') if logpdf is None else logpdf
        
        # Build the layers of the network
        self.layers = [self.parameters]
        self.weights = []
        self.biases = []
        for i in range(len(self.n_hidden)):
            with tf.variable_scope('nde_' + str(index) + '_layer_' + str(i + 1)):
                if i == 0:
                    self.weights.append(tf.get_variable("weights", [self.n_parameters, self.n_hidden[i]], initializer = tf.random_normal_initializer(0., np.sqrt(2./self.n_parameters))))
                    self.biases.append(tf.get_variable("biases", [self.n_hidden[i]], initializer = tf.constant_initializer(0.0)))
                elif i == len(self.n_hidden) - 1:
                    self.weights.append(tf.get_variable("weights", [self.n_hidden[i], self.N], initializer = tf.random_normal_initializer(0., np.sqrt(2./self.n_hidden[i]))))
                    self.biases.append(tf.get_variable("biases", [self.N], initializer = tf.constant_initializer(0.0)))
                else:
                    self.weights.append(tf.get_variable("weights", [self.n_hidden[i], self.n_hidden[i + 1]], initializer = tf.random_normal_initializer(0., np.sqrt(2/self.n_hidden[i]))))
                    self.biases.append(tf.get_variable("biases", [self.n_hidden[i + 1]], initializer = tf.constant_initializer(0.0)))
            if i < len(self.n_hidden) - 1:
                self.layers.append(self.activations[i](tf.add(tf.matmul(self.layers[-1], self.weights[-1]), self.biases[-1])))
            else:
                self.layers.append(tf.add(tf.matmul(self.layers[-1], self.weights[-1]), self.biases[-1]))

        # Map the output layer to mixture model parameters
        self.mu, self.sigma, self.alpha = tf.split(self.layers[-1], [self.M * self.n_data, self.M * self.n_data * (self.n_data + 1) // 2, self.M], 1)
        self.mu = tf.reshape(self.mu, (-1, self.M, self.n_data))
        self.sigma = tf.reshape(self.sigma, (-1, self.M, self.n_data * (self.n_data + 1) // 2))
        self.alpha = tf.nn.softmax(self.alpha)
        self.Sigma = tf.contrib.distributions.fill_triangular(self.sigma)
        self.Sigma = self.Sigma - tf.linalg.diag(tf.linalg.diag_part(self.Sigma)) + tf.linalg.diag(tf.exp(tf.linalg.diag_part(self.Sigma)))
        self.det = tf.reduce_prod(tf.linalg.diag_part(self.Sigma), axis=-1)

        self.mu = tf.identity(self.mu, name = "mu")
        self.Sigma = tf.identity(self.Sigma, name = "Sigma")
        self.alpha = tf.identity(self.alpha, name = "alpha")
        self.det = tf.identity(self.det, name = "det")
        
        # Log likelihoods
        self.L = tf.log(tf.reduce_sum(tf.exp(-0.5*tf.reduce_sum(tf.square(tf.einsum("ijlk,ijk->ijl", self.Sigma, tf.subtract(tf.expand_dims(self.data, 1), self.mu))), 2) + tf.log(self.alpha) + tf.log(self.det) - self.n_data*np.log(2. * np.pi) / 2.), 1) + 1e-37, name = "L")

        # Objective loss function
        self.trn_loss = -tf.reduce_mean(self.L, name = "trn_loss")

    def eval(self, xy, sess, log=True):
        """
        Evaluate log probabilities for given input-output pairs.
        :param xy: a pair (x, y) where x rows are inputs and y rows are outputs
        :param sess: tensorflow session where the graph is run
        :param log: whether to return probabilities in the log domain
        :return: log probabilities: log p(y|x)
        """
        
        x, y = xy
        lprob = sess.run(self.L,feed_dict={self.parameters:x,self.data:y})

        return lprob if log else np.exp(lprob)



