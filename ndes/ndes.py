import numpy as np
import numpy.random as rng
import tensorflow as tf
import ndes.mades

dtype = tf.float32

class BatchNormalizationExt(tf.layers.BatchNormalization):
    def eval_inv(self, sess, y):
        """
        Evaluates the inverse batch norm transformation for output y.
        :param y: output as numpy array
        :return: input as numpy array
        """
        gamma,beta,moving_mean,moving_variance = sess.run(self.variables)
        x_hat = (y - beta) / gamma
        x = np.sqrt(moving_variance) * x_hat + moving_mean

        return x

class ConditionalMaskedAutoregressiveFlow:
    """
    Implements a Conditional Masked Autoregressive Flow.
    """

    def __init__(self, n_inputs, n_outputs, n_hiddens, act_fun, n_mades, batch_norm=False, momentum=0.2,
                 output_order='sequential', mode='sequential', input=None, output=None, logpdf=None):
        """
        Constructor.
        :param n_inputs: number of (conditional) inputs
        :param n_outputs: number of outputs
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param act_fun: tensorflow activation function
        :param n_mades: number of mades in the flow
        :param batch_norm: whether to use batch normalization between mades in the flow
        :param momentum: momentum for moving mean and variance of the batch normalization layers
        :param output_order: order of outputs of last made
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        :param input: tensorflow placeholder to serve as input; if None, a new placeholder is created
        :param output: tensorflow placeholder to serve as output; if None, a new placeholder is created
        """

        # save input arguments
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hiddens = n_hiddens
        self.act_fun = act_fun
        self.n_mades = n_mades
        self.batch_norm = batch_norm
        self.momentum = momentum
        self.mode = mode

        self.input = tf.placeholder(dtype=dtype,shape=[None,n_inputs],name='x') if input is None else input
        self.y = tf.placeholder(dtype=dtype,shape=[None,n_outputs],name='y') if output is None else output
        self.logpdf = tf.placeholder(dtype=dtype,shape=[None],name='logpdf') if logpdf is None else logpdf
        self.training = tf.placeholder_with_default(False,shape=(),name="training")
        self.parms = []

        self.mades = []
        self.bns = []
        self.u = self.y
        self.logdet_dudy = 0.0

        for i in range(n_mades):

            # create a new made
            made = ndes.mades.ConditionalGaussianMade(n_inputs, n_outputs, n_hiddens, act_fun,
                                                 output_order, mode, self.input, self.u)
            self.mades.append(made)
            self.parms += made.parms
            output_order = output_order if output_order is 'random' else made.output_order[::-1]

            # inverse autoregressive transform
            self.u = made.u
            self.logdet_dudy += 0.5 * tf.reduce_sum(made.logp, axis=1,keepdims=True)

            # batch normalization
            if batch_norm:
                bn = BatchNormalizationExt(momentum=self.momentum)
                v_tmp = tf.nn.moments(self.u,[0])[1]
                self.u = bn.apply(self.u,training=self.training)
                self.parms += [bn.gamma,bn.beta]
                v_tmp = tf.cond(self.training,lambda:v_tmp,lambda:bn.moving_variance)
                self.logdet_dudy += tf.reduce_sum(tf.log(bn.gamma)) - 0.5 * tf.reduce_sum(tf.log(v_tmp+1e-5))
                self.bns.append(bn)

        self.output_order = self.mades[0].output_order

        # log likelihoods
        self.L = tf.add(-0.5 * n_outputs * np.log(2 * np.pi) - 0.5 * tf.reduce_sum(self.u ** 2, axis=1,keepdims=True), self.logdet_dudy,name='L')

        # train objective
        self.trn_loss = -tf.reduce_mean(self.L,name='trn_loss')
        self.trn_loss_reg = tf.abs(tf.reduce_mean(tf.subtract(self.L, self.logpdf)), name = "trn_loss_reg")

    def eval(self, xy, sess, log=True):
        """
        Evaluate log probabilities for given input-output pairs.
        :param xy: a pair (x, y) where x rows are inputs and y rows are outputs
        :param sess: tensorflow session where the graph is run
        :param log: whether to return probabilities in the log domain
        :return: log probabilities: log p(y|x)
        """
        
        x, y = xy
        lprob = sess.run(self.L,feed_dict={self.input:x,self.y:y})

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

        y = rng.randn(n_samples, self.n_outputs).astype(dtype) if u is None else u

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
        return sess.run(self.u,feed_dict={self.input:x,self.y:y})


class MixtureDensityNetwork:
    """
    Implements a Mixture Density Network for modeling p(y|x)
    """

    def __init__(self, n_inputs, n_outputs, n_components = 3, n_hidden=[50,50], activations=[tf.tanh, tf.tanh],
                 input=None, output=None, logpdf=None, batch_norm=False):
        """
        Constructor.
        :param n_inputs: number of (conditional) inputs
        :param n_outputs: number of outputs (ie dimensionality of distribution you're parameterizing)
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param activations: tensorflow activation functions for each hidden layer
        :param input: tensorflow placeholder to serve as input; if None, a new placeholder is created
        :param output: tensorflow placeholder to serve as output; if None, a new placeholder is created
        """
        
        # save input arguments
        self.D = n_outputs
        self.P = n_inputs
        self.M = n_components
        self.N = int((self.D + self.D * (self.D + 1) / 2 + 1)*self.M)
        self.n_hidden = n_hidden
        self.activations = activations
        self.batch_norm = batch_norm
        
        self.input = tf.placeholder(dtype=dtype,shape=[None,self.P],name='x') if input is None else input
        self.y = tf.placeholder(dtype=dtype,shape=[None,self.D],name='y') if output is None else output
        self.logpdf = tf.placeholder(dtype=dtype,shape=[None],name='logpdf') if logpdf is None else logpdf
        self.training = tf.placeholder_with_default(False,shape=(),name="training")
        
        # Build the layers of the network
        self.layers = [self.input]
        self.weights = []
        self.biases = []
        for i in range(len(self.n_hidden)):
            with tf.variable_scope('layer_' + str(i + 1)):
                if i == 0:
                    self.weights.append(tf.get_variable("weights", [self.P, self.n_hidden[i]], initializer = tf.random_normal_initializer(0., np.sqrt(2./self.P))))
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
        self.μ, self.Σ, self.α, self.det = self.mapping(self.layers[-1])
        self.μ = tf.identity(self.μ, name = "mu")
        self.Σ = tf.identity(self.Σ, name = "Sigma")
        self.α = tf.identity(self.α, name = "alpha")
        self.det = tf.identity(self.det, name = "det")
        
        # Log likelihoods
        self.L = tf.log(tf.reduce_sum(tf.exp(-0.5*tf.reduce_sum(tf.square(tf.einsum("ijlk,ijk->ijl", self.Σ, tf.subtract(tf.expand_dims(self.y, 1), self.μ))), 2) + tf.log(self.α) + tf.log(self.det) - self.D*np.log(2. * np.pi) / 2.), 1) + 1e-37, name = "L")

        # Objective loss function
        self.trn_loss = -tf.reduce_mean(self.L, name = "trn_loss")
        self.trn_loss_reg = -tf.reduce_mean(tf.subtract(self.L, self.logpdf), name = "trn_loss_reg")
        

    # Build lower triangular from network output (also calculate determinant)
    def lower_triangular_matrix(self, σ):
        Σ = []
        det = []
        start = 0
        end = 1
        for i in range(self.D):
            exp_val = tf.exp(σ[:, :, end-1])
            det.append(exp_val)
            if i > 0:
                Σ.append(tf.pad(tf.concat([σ[:, :, start:end-1], tf.expand_dims(exp_val, -1)], -1), [[0, 0], [0, 0], [0, self.D-i-1]]))
            else:
                Σ.append(tf.pad(tf.expand_dims(exp_val, -1), [[0, 0], [0, 0], [0, self.D-i-1]]))
            start = end
            end += i + 2
        Σ = tf.transpose(tf.stack(Σ), (1, 2, 0, 3))
        det = tf.reduce_prod(tf.stack(det), 0)
        return Σ, det

    # Split network output into means, covariances and weights (also returns determinant of covariance)
    def mapping(self, output_layer):
        μ, Σ, α = tf.split(output_layer, [self.M * self.D, self.M * self.D * (self.D + 1) // 2, self.M], 1)
        μ = tf.reshape(μ, (-1, self.M, self.D))
        Σ, det = self.lower_triangular_matrix(tf.reshape(Σ, (-1, self.M, self.D * (self.D + 1) // 2)))
        α = tf.nn.softmax(α)
        return μ, Σ, α, det


    def eval(self, xy, sess, log=True):
        """
        Evaluate log probabilities for given input-output pairs.
        :param xy: a pair (x, y) where x rows are inputs and y rows are outputs
        :param sess: tensorflow session where the graph is run
        :param log: whether to return probabilities in the log domain
        :return: log probabilities: log p(y|x)
        """
        
        x, y = xy
        lprob = sess.run(self.L,feed_dict={self.input:x,self.y:y})

        return lprob if log else np.exp(lprob)

