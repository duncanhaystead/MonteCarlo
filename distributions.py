import numpy as np
import math

class Distribution:
    def __init__(self, a = -1, b = 1, *args, **kwargs):
        if kwargs:
                #keyword arguments provided
            if ("normal" in kwargs):
                self.type = 'normal'
                if ("samples" in kwargs):
                    self.samples = kwargs.get("data", [])
                    self.get_stats()
                else:
                    self.mu = kwargs.get("mean", 0.0)
                    self.sd = kwargs.get('sd', 1.0)
                    #get mean and sd; default to standard normal value if not given
                self.f = np.random.normal(loc = self.mu, scale = self.sd)
            elif ("custom" in kwargs):
                self.type = 'custom'
                self.f = kwargs.get("custom", x)
                self.ndim = kwargs.get("dim", 1.0)
                if ("samples" in kwargs):
                    self.samples = kwargs.get("data", [])
                    if self.samples:
                        self.gen_samples()
                        self.get_stats()
                    else:
                        self.mu = self.mu = kwargs.get("mean", 0.0)
                        self.sd = kwargs.get('sd', 1.0)
            else:
                raise Error("no keyword arguments provided")

            self.a = a
            self.b = b
            self.nsamples = 1000

    def generate_distribution(self, p, a, b,ndim=1, mu=None, sigma=None,num_samples=1000):
        distribution = Distribution(p, a, b, ndim, mu, sigma, num_samples)
        return distribution

    def BoxMuller(self):
        xi, eta = np.random.uniform(), np.random.uniform()
        return self.mu + self.sigma*np.sqrt(-2*np.log(eta))*math.cos(2*math.pi*xi)

    def get_stats(self):
        self.mean, self.sd = np.mean(self.samples), np.std(self.samples)

    def gen_samples(self, n = 1000):
        if self.ndim == 1:
            self.samples = [self.f(c) for c in np.random.uniform(self.a, self.b,size = n)]
        else:
            self.samples = [self.f(*c) for c in np.random.uniform(self.a, self.b,size = (n, self.ndim))]
        self.get_stats()

    def pdf(self, y):
        curr_expected_value = sum([abs(x) for x in self.samples])
        measure = 1 - curr_expected_value/y
        return measure/self.nsamples

    def append_sample(self, sample):
        self.samples.append(sample)
        self.nsamples +=1
        self.get_stats()

    def rvs(self):
        if self.normal:
            return self.BoxMuller()
        i = np.random.random_integers(0, self.nsamples-1)
        return self.samples[i]
