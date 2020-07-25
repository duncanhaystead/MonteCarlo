from distributions import *
import numpy as np
from math import e


class MonteCarlo:
        def __init__(self, *args, **kwargs):
            self.a = a
            self.b = b
            self.ndim = ndim
            self.sigma = sigma
            self.mu = mu
            self.p = p
            self.q = q

        def get_sampler(self, sample_method):
            return self.methods(sample_method)

        def Uniform(self, q= None, N = 1000):
            p = self.p
            p.gen_samples(n = N)
            return p.samples

        def AcceptanceRejection(self, p = None, q = None, N= 1000, C = None, nsamples_target = 5000):
            if not p:
                p = self.p
            if not p.nsamples:
                p.nsamples = nsamples_target
            if not q:
                q = self.q
            q.gen_samples()
            p.gen_samples()
            if not C:
                q_min = min(list(filter(lambda x: abs(x) != 0, q.samples)))
                p_max = max(list(filter(lambda x: abs(x) != 0, p.samples)))
                C = abs(p_max/q_min)
            X = []
            while len(X) < N:
                x = q.rvs()
                u = np.random.uniform(0,1)
                try:
                    alpha = p.pdf(x)/(C*q.pdf(x))
                except ZeroDivisionError:
                    alpha = 0
                if u < alpha:
                    X.append(x)
            return X

        def Metropolis(self, p=None,q=None, mu = 0, sigma = 1, N = 1000, x0 = None):
            if not p:
                p = self.p
            if not q:
                q = self.q
            q.gen_samples()
            if not x0:
                x0 = q.rvs()
            X = [x0]
            p.append_sample(x0)
            while len(X) < N:
                x = X[-1]
                y = q.pdf(x) + q.rvs()
                u = np.random.uniform(0,1)
                try:
                    alpha = min(1,p.pdf(y)/p.pdf(x))
                except ZeroDivisionError:
                    alpha = 0
                if u <= alpha:
                    X.append(y)
                    X.append(y)
                else:
                    X.append(x)
                    X.append(x)
            return X

        def MetropolisHastings(self,p=None, q=None, mean = 0,std = 1, N = 1000, x0 = None):
            if not p:
                p = self.p
            if not q:
                q  = self.q
            q.gen_samples()
            if not x0:
                x0 = q.rvs()
            X = [x0]
            p.append_sample(x0)
            while len(X) < N:
                x = X[-1]
                y = q.pdf(x) + q.rvs()
                u = np.random.uniform(0,1)
                try:
                    alpha = min(1,(p.pdf(y)*q.pdf(x-y))/(p.pdf(x)*p.pdf(y-x)))
                except ZeroDivisionError:
                    alpha = 0
                if u <= alpha:
                    X.append(y)
                    p.append_sample(y)
                else:
                    X.append(x)
                    p.append_sample(x)
            return X

        def methods(self,method_name):
            return {
              'Uniform': self.Uniform,
              'AcceptanceRejection':  self.AcceptanceRejection,
              'Metropolis':  self.Metropolis,
              'MetropolisHastings': self.MetropolisHastings
              }.get(method_name, "Method %s not implemented" % method_name)
