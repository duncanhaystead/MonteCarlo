import numpy as np
import math

class Distribution:
    def __init__(self, f, a, b, ndim = 1, normal = False, mu=None, sigma=None, samples = [], nsamples=1000):
        self.f = f
        self.a = a
        self.b = b
        self.ndim = ndim
        self.nsamples = nsamples
        self.samples = samples
        self.normal = normal
        self.mu = mu
        self.sigma = sigma

    def generate_distribution(self, p, a, b,ndim=1, mu=None, sigma=None,num_samples=1000):
        distribution = Distribution(p, a, b, ndim, mu, sigma, num_samples)
        return distribution

    def BoxMuller(self):
        xi, eta = np.random.uniform(), np.random.uniform()
        return self.mu + self.sigma*np.sqrt(-2*np.log(eta))*math.cos(2*math.pi*xi)

    def get_stats(self):
        self.mu, self.sigma = np.mean(self.samples), np.std(self.samples)

    def gen_samples(self, n = 1000):
        if self.ndim == 1:
            self.samples = [self.f(c) for c in np.random.uniform(self.a, self.b,size = n)]
        else:
            self.samples = [self.f(*c) for c in np.random.uniform(self.a, self.b,size = (n, self.ndim))]
        self.get_stats()

    def pdf(self, x):
        measure = sum([1 if y >= abs(x) else 0 for y in self.samples])
        return np.log(measure/self.nsamples)

    def append_sample(self, sample):
        self.samples.append(sample)
        self.nsamples +=1
        self.get_stats()

    def rvs(self):
        if self.normal:
            return self.BoxMuller()
        i = np.random.random_integers(0, self.nsamples-1)
        return self.samples[i]
