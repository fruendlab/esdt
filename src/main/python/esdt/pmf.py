import numpy as np
from scipy import stats


class PsychometricFunction(object):

    def __init__(self, data):
        self.data = data

    def psi(self, stimulus, parameters):
        raise NotImplementedError

    def get_thres(self, param, prob=0.75):
        raise NotImplementedError

    def get_slope(self, param, prob=0.75):
        raise NotImplementedError

    @property
    def priors(self):
        raise NotImplementedError

    def loglikelihood(self, param):
        prob = self.psi(self.data[:, 0], param)
        B = stats.binom(self.data[:, 2].reshape((-1, 1)), prob)
        out = B.logpmf(self.data[:, 1].reshape((-1, 1)))
        return out.sum(0)

    def posterior(self, param):
        p = self.loglikelihood(param)
        for prior, par in zip(self.priors, param):
            p += prior.logpdf(par)
        p -= p.max()
        np.exp(p, p)
        return p


class LogisticAfc(PsychometricFunction):

    def __init__(self, data, guessing=1./2):
        super(LogisticAfc, self).__init__(data)
        self.guessing = guessing
        self.priors_ = [
            stats.norm(0, scale=100),
            stats.norm(0, scale=100),
            stats.beta(1.5, 20),
        ]

    def psi(self, x, param):
        x = np.asarray(x).reshape((-1, 1))
        lm = param[2].reshape((1, -1))
        F = 1.+np.exp(- param[0].reshape((1, -1))
                      - param[1].reshape((1, -1))*x)
        return self.guessing + (1 - self.guessing - lm)/F

    def get_thres(self, param, prob=0.75):
        q = (prob - self.guessing)/(1 - self.guessing - param[2])
        logit = np.log(q/(1-q))
        return (logit - param[0])/param[1]

    def get_slope(self, param, prob=0.75):
        thres = self.get_thres(param, prob=prob)
        logistic = 1./(1+np.exp(-param[0] - param[1]*thres))
        return (1 - self.guessing - param[2])*logistic*(1-logistic)*param[1]

    priors = property(fget=lambda self: self.priors_)


def mkgrid(*ranges):
    ranges = [slice(start, end, 1j*steps) for start, end, steps in ranges]
    return np.array(np.mgrid[tuple(ranges)])


def integrate_posterior(pmf, grid):
    grid.shape = (grid.shape[0], -1)
    posterior = pmf.posterior(grid)
    Z = posterior.sum()
    posterior /= Z
    return grid, posterior


def get_stats(posterior, stat):
    E1 = np.dot(posterior, stat)
    E2 = np.dot(posterior, stat**2)
    return E1, np.sqrt(E2 - E1**2)
