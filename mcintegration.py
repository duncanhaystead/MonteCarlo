import sys
sys.path.append('/Users/duncanhaystead/Desktop/coding/mcintegrate/')
from distributions import Distribution
from montecarlo import MonteCarlo
import numpy as np
from numpy import linalg as LA
from scipy.stats import norm
import scipy.stats as st
import matplotlib.pyplot as plt

class MonteCarloIntegration(MonteCarlo):
    def __init__(self, f, a, b, ndim=1, n_distsamples=1000):
        MonteCarlo.__init__(self)
        self.p = Distribution(f, a, b, ndim, nsamples = n_distsamples)
        self.a = a
        self.b = b
        self.ndim = ndim

    def integrate(self, proposal, nsamples = 1000, burn_in = 1000, sample_method = 'Uniform',
    proposal_is_distribution = False, nsamples_proposal = 1000, normal_proposal = False,
    proposal_mu = None, proposal_sigma = None, diagnostics = True):
        if proposal_is_distribution:
            self.q = proposal
        else:
            self.q = Distribution(proposal, self.a, self.b, self.ndim, nsamples = nsamples_proposal,
            normal = normal_proposal, mu = proposal_mu, sigma = proposal_sigma)
        sampler = self.get_sampler(sample_method)
        samples = sampler(N = nsamples + burn_in)[burn_in:]
        plot = plt.plot(samples)
        if self.ndim > 1:
            v = 1
            zip_lst = zip(self.b, self.a)
            for a,b in zip_lst:
                v *= abs(b-a)
        else:
            v = abs(self.b - self.a)
        integral = np.sum([x for x in samples])/(nsamples)*v
        error = self.std_error(samples, nsamples)*v
        return integral, error, plot


    def std_error(self, vals, n):
        sq_sum = np.sum([x**2 for x in vals])/n
        sum = np.sum([abs(x) for x in vals])/n
        error = np.sqrt(sq_sum - sum**2)/np.sqrt(n)
        return error
