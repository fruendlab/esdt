from collections import namedtuple
import numpy as np
from scipy import stats, optimize
try:
    import pylab as pl
    HASPYLAB = True
except ImportError:
    HASPYLAB = False

from .utils import safe_log


# Sigmoid functions
def gumbel(x, th, w):
    def z(al):
        return np.log(-np.log(al))

    y = (z(0.1) - z(0.9))*(x-th)/w + z(0.5)
    return 1-np.exp(-np.exp(y))


def logistic(x, th, w):
    def z(al):
        return 2*np.log(1/al-1)

    y = z(0.1)*(x-th)/w
    return 1./(1+np.exp(-y))


class PsychometricFunction(object):

    def __init__(self, F, guess=None):
        self.F = F
        self.start = None
        self.params = None
        self.sem = None
        self.param_r = None
        self.loglikelihood = None
        self.priors = ()
        self.guess = guess
        self.data = None

    def fit(self, data, assign=True, start=None):
        self.data = data
        x, k, n, y = self.extract_data()

        def loss(params):
            return (self.negloglikelihood(params, x, k, n)
                    + 1e7*np.logical_or(params[-1] > 0.1, params[-1] < 0)
                    + 1e7*np.logical_or(params[0] < x.min(),
                                        params[0] > x.max()))

        if start is None:
            start = self.get_start(x, y)

        params = optimize.fmin(loss, start, disp=False)
        loglikelihood = self.negloglikelihood(params, x, k, n)
        if assign:
            self.start = start
            self.params = params
            self.loglikelihood = loglikelihood
        return params, loglikelihood

    def negloglikelihood(self, params, x, k, n):
        p = self.predict(x, params)
        return np.sum(-k*safe_log(p) - (n-k)*safe_log(1-p))

    def posterior(self, params):
        d = self.extract_data()
        ll = np.zeros(params.shape[1], 'd')
        for i in range(params.shape[1]):
            ll[i] = self.negloglikelihood(params[:, i], d.x, d.k, d.n)
        for par, prior in zip(params, self.priors):
            ll += prior.logpmf(par)
        ll -= ll.max()  # To avoid overflow, posterior is unnormalized anyways
        np.exp(ll, ll)
        return ll

    def extract_data(self):
        Dataset = namedtuple('Dataset', ['x', 'k', 'n', 'y'])
        return Dataset(self.data[:, 0],
                       self.data[:, 1]*self.data[:, 2],
                       self.data[:, 2],
                       self.data[:, 1])

    def jackknife_sem(self, data):
        params = []
        for i in range(data.shape[0]):
            mask = np.ones(data.shape[0], bool)
            mask[i] = False
            params.append(self.fit(data[mask],
                                   assign=False,
                                   start=self.params)[0])
        v = (len(params)-1)*np.var(np.array(params), 0)
        r = np.corrcoef(np.array(params).T)
        se = np.sqrt(v/len(params))
        self.sem = se
        self.param_r = r
        return se, r

    def predict(self, x, params=None):
        if params is None:
            params = self.params
        if self.guess is None:
            guess = params[0]
            params = params[1:]
        else:
            guess = self.guess
        return guess + (1-guess-params[-1])*self.F(x, params[0], params[1])

    def get_start(self, x, y):
        if self.guess is None:
            guess = 0.02
        else:
            guess = self.guess
        y_ = (y-guess)/(1-guess)
        y = y_
        i = np.logical_and(y > 0, y < 1)
        y = y[i]
        logit = np.log(y/(1-y))
        slope, intercept = stats.linregress(x[i], logit)[:2]
        th = -intercept/slope
        w = 2/slope
        if self.guess is None:
            return np.array([th, w, 0.02, 0.02])
        else:
            return np.array([th, w, 0.02])


def bayesian_inference(pmf, statistics={'threshold': lambda grid: grid[0],
                                        'width': lambda grid: grid[1]}):
    grid = mkgrid(
        *[(par - 3*se, par + 3*se, 20) for par, se in zip(pmf.params, pmf.sem)]
    )
    grid, posterior = integrate_posterior(pmf, grid)
    return {key: get_stats(posterior, getter(grid))
            for key, getter in statistics.items()}


def pmfplot(pmf, **kwargs):
    if 'axes' in kwargs:
        ax = kwargs.pop('axes')
    else:
        if HASPYLAB:
            ax = pl.axes()
        else:
            raise RuntimeError(
                'Trying to use pylab function by pylab does not exist')
    col = kwargs.setdefault('color', 'black')

    dataset = pmf.extract_data()
    xmin = dataset.x.min()
    xmax = dataset.x.max()
    x = kwargs.pop('x', np.mgrid[xmin:xmax:100j])
    i0 = x <= xmin
    i1 = x >= xmax
    i2 = np.logical_and(x >= xmin, x <= xmax)

    p = pmf.predict(x)
    se = determine_error_region(pmf, x)
    ax.fill_between(x, p+se, p-se, facecolor=col, alpha=0.5)
    ax.plot(x[i0], p[i0], '--', color=col)
    ax.plot(x[i1], p[i1], '--', color=col)
    ax.plot(x[i2], p[i2], **kwargs)
    ax.scatter(dataset.x, dataset.y, s=dataset.n, c=col)


def determine_error_region(pmf, x, n=50):
    C = pmf.param_r * np.outer(pmf.sem, pmf.sem)
    params = np.random.multivariate_normal(pmf.params, C, size=n)
    pred = np.array([pmf.predict(x, par) for par in params])
    return pred.std(0)


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
