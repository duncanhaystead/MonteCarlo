import numpy as np
from scipy.stats import norm
import scipy.stats as st

class MCIntegration():


    def __init__(self, func, lower, upper, ndim):
        self.f = func
        self.a = lower
        self.b = upper
        self.ndim = ndim
        # self.proposal =

    def integrate(self, Markov = False, method = 'Naive', N = 1000, trials = 'fixed', diagnostics = 'on', proposal = None, mean= None, sd=None):
        # if proposal is None:
        #     if n = 1:
        #         self.proposal = np.random.uniform(a, b, N)
        #     else:
        MC = MonteCarlo(self.a, self.b, self.ndim, sd, mean, self.f)
        if Markov:
            MCMC = MarkovChainMonteCarlo(MC.__init__())
            method = getattr(MCMC, method)
            vals = method(N)
            integral = np.mean(vals)*(self.b-self.a)
            return integral
        method = getattr(MC, method)
        vals = method(N)
        mean_f = np.mean([self.f(x) for x in vals])
        integral = mean_f*(self.b-self.a)
        return integral
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


class MonteCarlo():

    def __init__(self,lower=0,upper=1,ndim=1,sigma=1, mu = 0, func = None, pdf=None):
        self.pdf = pdf
        self.a = lower
        self.b = upper
        self.ndim = ndim
        self.sigma = sigma
        self.mu = mu
        self.f = func

    def Naive(self, N = 1000):
        if not self.f:
            samples = [np.random.uniform(self.a, self.b, N)]
        else:

            samples = [self.f(x) for x in np.random.uniform(self.a, self.b, N)]
        return samples

    def Rejection(self, N= 1000, p = None, q = None):
        if not self.f:
            p = lambda x : st.norm.pdf(x, loc=30, scale=10) + st.norm.pdf(x, loc=80, scale=20)
        else:
            p = self.f
        q = lambda x :st.norm.pdf(x, loc=self.mu, scale=self.sigma)
        # if q:
        #     q = proposal_pdf
        # if p:
        #     p = target_pdf
        X = np.zeros(N)
        y = np.linspace(self.a, self.b, N)
        C = max(p(y)/q(y))
        else:
            raise NotImplementedError("This method to find the maximum is not implemented")
        for i in range(N):
            x = st.norm.rvs(loc = self.mu, scale=self.sigma)
            u = np.random.uniform(0,1)
            if u <= p(x)/(C*q(x)):
                X[i] = x
            else:
                X = np.delete(X, i)
        return X

class MarkovChainMonteCarlo(MonteCarlo):

    def __init__(self,lower=0,upper=1,ndim=1,sigma=1, mu = 0, func = None, pdf=None):
        self.pdf = pdf
        self.a = lower
        self.b = upper
        self.ndim = ndim
        self.sigma = sigma
        self.mu = mu
        self.f = func

    def Metropolis(self, N = 1000, q = None, p = None):
        if not self.f:
            p = lambda x : st.norm.pdf(x, loc=30, scale=10) + st.norm.pdf(x, loc=80, scale=20)
        else:
            p = self.f
        # if q:
        #     q = proposal_pdf
        # if p:
        #     p = target_pdf
        q = lambda x :st.norm.pdf(x, loc=self.mu, scale=self.sigma)
        X = np.zeros(N)
        X[0] = st.norm.rvs(loc = self.mu, scale=self.sigma)
        for i in range(N-1):
            x = X[i]
            y = q(x) + st.norm.rvs(loc = self.mu, scale=self.sigma)
            u = np.random.uniform(0,1)
            alpha = min(1,p(y)/p(x))
            if u <= alpha:
                X[i+1] = y
            else:
                X[i+1] = x
        return X
#
    def MetropolisHastings(self, N = 1000, q = None, p = None):
        if not self.f:
            p = lambda x : st.norm.pdf(x, loc=30, scale=10) + st.norm.pdf(x, loc=80, scale=20)
        else:
            p = self.f
        q = lambda x :st.norm.pdf(x, loc=self.mu, scale=self.sigma)
        X = np.zeros(N)
        X[0] = st.norm.rvs(loc = self.mu, scale=self.sigma)
        for i in range(N-1):
            x = X[i]
            y = q(x) + st.norm.rvs(loc = self.mu, scale=self.sigma)
            u = np.random.uniform(0,1)
            alpha = min(1,(p(y)*q(x-y))/(p(x)*q(y-x)))
            if u <= alpha:
                X[i+1] = y
            else:
                X[i+1] = x
        return X
#
#     def Gibbs(self, f = None):
