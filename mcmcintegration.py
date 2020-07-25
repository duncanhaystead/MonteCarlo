import numpy as np
from scipy.stats import norm
import scipy.stats as st
from montecarlo.py import MonteCarlo

class MonteCarloIntegration():


    def __init__(self, f, lower=-1, upper=1, ndim=1):
        if not f:
            raise Exception("No function has been given")
        self.f = f
        self.a = lower
        self.b = upper
        self.ndim = ndim
        self.mc = MonteCarlo(lower = self.a, upper = self.b,ndim= self.ndim, target = distribution_gen('f_pdf'))
        # self.proposal =
        class distribution_gen(st.rv_continuous(self.a, self.b)):
            "Generate a probability distribution for a given function"
            def _pdf(self, x):
                return self.f(x)

    def integrate(self, proposal = None, mu = None, sigma = None, N = 1000, initial_guess=None,
        diags_on = True):
        if not mu or not sigma:
            mu, sigma = get_stats(proposal, N)
        if not initial_guess:
            initial_guess = (self.a + self.b)/2
        method = getattr(MC, method)
        vals = method(N)
        mean_f = np.mean([self.f(x) for x in vals])
        integral = mean_f*(self.b-self.a)
        return integral

    def get_stats(self, p, N):
        X = np.random.uniform(self.a, self.b, N)
        samples = [p(x) for x in X]
        return np.mean(samples), np.std(samples)

        # points = self.method(N)
        # mean, errorterm = f_mean(points)
        # integral = 1/N*mean*self.measure()
        # error = np.sqrt(errorterm/self.n - integral**2)/np.sqrt(self.n)
        # if diagnostics = 'on':
        #     return integral, self.run_diagnostics(integral)
        # return integral, error

    # def f_mean(points):
    #     values = [self.f(x.flatten) for x in points]
    #     return np.sum(values), np.sum([f**2 for f in values])
    #
    #
    # def measure(self):
    #     return np.multiply([b-a for a, b in self.a, self.b])


  #
  #   def run_diagnostics(self, integral):
