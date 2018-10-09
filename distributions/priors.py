from scipy.stats import multivariate_normal
import numpy as np

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

    def draw(self):

        P = 0
        while P == 0:
            x = self.mean + np.dot(self.L, np.random.normal(0, 1, len(self.mean)))
            P = self.uniform(x)
        return x

    def pdf(self, x):

        return self.uniform(x)*multivariate_normal.pdf(x, mean=self.mean, cov=self.C)
