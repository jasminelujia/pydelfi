import numpy as np
import numpy.random as rng

def discrete_sample(p, n_samples=1):
    """
    Samples from a discrete distribution.
    :param p: a distribution with N elements
    :param n_samples: number of samples
    :return: vector of samples
    """

    # check distribution
    #assert isdistribution(p), 'Probabilities must be non-negative and sum to one.'

    # cumulative distribution
    c = np.cumsum(p[:-1])[np.newaxis, :]

    # get the samples
    r = rng.rand(n_samples, 1)
    return np.sum((r > c).astype(int), axis=1)


class MarkovJumpProcess:
    """Implements a generic markov jump process and algorithms for simulating it.
    It is an abstract class, it needs to be inherited by a concrete implementation."""

    def __init__(self, init, params):

        self.state = np.asarray(init)
        self.params = np.asarray(params)
        self.time = 0.0

    def _calc_propensities(self):
        raise NotImplementedError('This is an abstract method and should be implemented in a subclass.')

    def _do_reaction(self, reaction):
        raise NotImplementedError('This is an abstract method and should be implemented in a subclass.')

    def sim_steps(self, num_steps):
        """Simulates the process with the gillespie algorithm for a specified number of steps."""

        times = [self.time]
        states = [self.state.copy()]

        for _ in range(num_steps):

            rates = self.params * self._calc_propensities()
            total_rate = rates.sum()

            if total_rate == 0:
                self.time = float('inf')
                break

            self.time += rng.exponential(scale=1/total_rate)

            reaction = discrete_sample(rates / total_rate)[0]
            self._do_reaction(reaction)

            times.append(self.time)
            states.append(self.state.copy())

        return times, np.array(states)

    def sim_time(self, dt, duration, max_n_steps=float('inf')):
        """Simulates the process with the gillespie algorithm for a specified time duration."""

        num_rec = int(duration / dt) + 1
        states = np.zeros([num_rec, self.state.size])
        cur_time = self.time
        n_steps = 0

        for i in range(num_rec):

            while cur_time > self.time:

                rates = self.params * self._calc_propensities()
                total_rate = rates.sum()

                if total_rate == 0:
                    self.time = float('inf')
                    break

                self.time += rng.exponential(scale=1/total_rate)

                reaction = discrete_sample(rates / total_rate)[0]
                self._do_reaction(reaction)

                n_steps += 1
                if n_steps > max_n_steps:
                    raise SimTooLongException(max_n_steps)

            states[i] = self.state.copy()
            cur_time += dt

        return np.array(states)


class LotkaVolterra(MarkovJumpProcess):
    """Implements the lotka-volterra population model."""

    def _calc_propensities(self):

        x, y = self.state
        xy = x * y
        return np.array([xy, x, y, xy])

    def _do_reaction(self, reaction):

        if reaction == 0:
            self.state[0] += 1
        elif reaction == 1:
            self.state[0] -= 1
        elif reaction == 2:
            self.state[1] += 1
        elif reaction == 3:
            self.state[1] -= 1
        else:
            raise ValueError('Unknown reaction.')


def calc_summary_stats(states):
    """
        Given a sequence of states produced by a simulation, calculates and returns a vector of summary statistics.
        Assumes that the sequence of states is uniformly sampled in time.
        """
    
    N = states.shape[0]
    x, y = states[:, 0].copy(), states[:, 1].copy()
    
    # means
    mx = np.mean(x)
    my = np.mean(y)
    
    # variances
    s2x = np.var(x, ddof=1)
    s2y = np.var(y, ddof=1)
    
    # standardize
    x = (x - mx) / np.sqrt(s2x)
    y = (y - my) / np.sqrt(s2y)
    
    # auto correlation coefficient
    acx = []
    acy = []
    for lag in [1, 2]:
        acx.append(np.dot(x[:-lag], x[lag:]) / (N-1))
        acy.append(np.dot(y[:-lag], y[lag:]) / (N-1))

    # cross correlation coefficient
    ccxy = np.dot(x, y) / (N-1)

    return np.array([mx, my, np.log(s2x + 1), np.log(s2y + 1)] + acx + acy + [ccxy])
