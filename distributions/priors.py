from scipy.stats import multivariate_normal
import numpy as np

class TruncatedGaussian():

    def __init__(self, mean, C, lower, upper):

        self.mean = mean
        self.C = C
        self.Cinv = np.linalg.inv(C)
        self.lower = lower
        self.upper = upper
        self.L = np.linalg.cholesky(C)
        self.logdet = np.log(np.linalg.det(C))

    def loguniform(self, x):

        inrange = np.prod(x > self.lower)*np.prod(x < self.upper)
        return inrange*np.log(np.prod(self.upper-self.lower)) - (1 - inrange)*1e300

    def uniform(self, x):

        inrange = np.prod(x > self.lower)*np.prod(x < self.upper)
        return inrange*np.prod(self.upper-self.lower)


    def draw(self, to_draw = 1):
        x_keep = None
        if to_draw == 1:
            squeeze = True
        else:
            squeeze = False
        while to_draw > 0:
            cov = np.random.normal(0, 1, (to_draw, self.mean.shape[0]))
            x = self.mean[np.newaxis, :] + np.dot(cov, self.L)
            up = x <= self.upper[np.newaxis, :]
            down = x >= self.lower[np.newaxis, :]
            ind = np.argwhere(np.all(up * down, axis = 1))[:, 0]
            if x_keep is None:
                x_keep = x[ind]
            else:
                x_keep = np.concatenate([x_keep, x[ind]])
            to_draw -= ind.shape[0]
        if squeeze:
            x_keep = np.squeeze(x_keep, 0)
        return x_keep

    def pdf(self, x):

        return np.exp(self.logpdf(x))

    def logpdf(self, x):

        return self.loguniform(x) - 0.5*self.logdet - 0.5*np.dot((x - self.mean), np.dot(self.Cinv,(x - self.mean)) )


class Uniform():

    def __init__(self, lower, upper):

        self.lower = lower
        self.upper = upper

    def logpdf(self, x):

        inrange = np.prod(x > self.lower)*np.prod(x < self.upper)
        return inrange*np.log(np.prod(self.upper-self.lower)) - (1 - inrange)*np.inf

    def pdf(self, x):

        inrange = np.prod(x > self.lower)*np.prod(x < self.upper)
        return inrange*np.prod(self.upper-self.lower)

    def draw(self):

        return np.random.uniform(self.lower, self.upper)
