import numpy as np
from scipy.stats import multivariate_normal

class TruncatedGaussian():

    def __init__(self, mean, C, lower, upper):

        self.mean = mean
        self.C = C
        self.lower = lower
        self.upper = upper
        self.L = np.linalg.cholesky(C)
        
    def uniform(self, x):

        # Test each parameter against prior limits
        for a in range(0, len(x)):
    
            if x[a] > self.upper[a] or x[a] < self.lower[a]:
                return 0
        return np.prod(self.upper-self.lower)

    #def draw(self, to_draw = 1):
    #
    #    P = 0
    #    while P == 0:
    #        x = self.mean + np.dot(self.L, np.random.normal(0, 1, len(self.mean)))
    #        P = self.uniform(x)
    #    return x
    
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

        return self.uniform(x)*multivariate_normal.pdf(x, mean=self.mean, cov=self.C)