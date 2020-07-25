import numpy as np
from scipy.stats import norm
import scipy.stats as st

class MonteCarlo:
        def __init__(self,lower=0,upper=1,ndim=1,sigma=1, mu = 0, proposal = None):
            if proposal:
                self.q = proposal
            else:
                self.q = lambda x : st.norm.pdf(x, loc=mu, scale=sigma)
            self.a = lower
            self.b = upper
            self.ndim = ndim
            self.sigma = sigma
            self.mu = mu
            self.f = fun

        def log_transform(self, probabilities):
            return np.log(probabilities)

        def Naive(self, N = 1000, q = None):
            if not q:
                q = self.q
            samples = [np.random.uniform(self.a, self.b, N)]
            else:

                samples = [self.f(x) for x in np.random.uniform(self.a, self.b, N)]
            return samples

        def Rejection(self, N= 1000, p = None, q = None):
            if not q:
                q = self.q
            q = lambda x :st.norm.pdf(x, loc=self.mu, scale=self.sigma)
            # if q:
            #     q = proposal_pdf
            # if p:
            #     p = target_pdf
            X = np.zeros(N)
            y = np.linspace(self.a, self.b, N)
            C = max(p(y)/q(y))
            for i in range(N):
                x = st.norm.rvs(loc = self.mu, scale=self.sigma)
                u = np.random.uniform(0,1)
                if u <= p(x)/(C*q(x)):
                    X[i] = x
                else:
                    X = np.delete(X, i)
            return X

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
    def MetropolisHastings(self, N = 1000, q = None, p = None, initial_guess = None):
        if not q:
            q = self.q
        if not p:
            p = self.p
        q = lambda x :st.norm.pdf(x, loc=self.mu, scale=self.sigma)
        if not initial_guess:
            X[0] = st.norm.rvs(loc = self.mu, scale=self.sigma)
        else:
            X[0] = initial_guess
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
