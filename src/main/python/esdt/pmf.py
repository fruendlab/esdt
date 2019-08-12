from collections import namedtuple
import numpy as np
from scipy import stats, optimize
import warnings
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

    def __init__(self, F, guess=None, transform=None):
        """Main psychometrc function model

        Args:
            F(x, th, w):
                Sigmoid to be used. This can be any function that takes an
                array of x values as the first argument and a threshold and
                width as second and third arguments.
            guess:
                Guessing rate. If this is None, the guessing rate will be
                estimated otherwise it will be fixed at the given value.
                Use guess=0.5 for 2afc for example.
            transform:
                Transformation to apply to stimulus intensities before fitting.
                For example, if F=gumbel and transform=np.log10, the resulting
                psychometric function would effectively fit a Weibull function.
        """
        self.F = F
        self.start = None
        self.params = None
        self.sem = None
        self.param_r = None
        self.loglikelihood = None
        self.priors = ()
        self.guess = guess
        self.data = None
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

    def ml_fit(self, data, assign=True, start=None):
        """Maximum likelihood fit to data

        Args:
            data:
                an array with one block of trials per row. First column should
                be stimulus intensity, second column should be the fraction of
                correct trials and the third colunn should be the total number
                of trials in the respective block.
            assign:
                Should the estimated parameters be assigned to the PMF object?
            start:
                Starting values to use for the fit. If this is None, starting
                values will be determined heuristically using linear regression

        Returns:
            params, -loglikelihood
            params:
                parameter estimate of the psychometric function model.
                Interpretation of these depends on how you parameterized your
                model. Order of parameters is (guess), th, w, lapse, where
                guess is the guessing rate (if specified), th and w are the
                parameters used for the sigmoid function F (see constructor),
                and lapse is the lapse rate/ upper asymptote of the
                psychometric function.
            -loglikelihood:
                In fact, this is the negative log-likelihood of the data under
                the model.
        """
        if assign:
            self.data = data

        x, k, n, y = self.extract_data(data)

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

    def fit(self, *args, **kwargs):
        warnings.warn(
            'The fit() method is deprecated. Use ml_fit() instead.',
            DeprecationWarning)
        return self.ml_fit(*args, **kwargs)

    def negloglikelihood(self, params, x, k, n):
        p = self.predict(x, params)
        return np.sum(-k*safe_log(p) - (n-k)*safe_log(1-p))

    def posterior(self, params):
        d = self.extract_data()
        ll = np.zeros(params.shape[1], 'd')
        for i in range(params.shape[1]):
            ll[i] = self.negloglikelihood(params[:, i], d.x, d.k, d.n)
        for par, prior in zip(params, self.priors):
            ll += prior.logpdf(par)
        ll -= ll.max()  # To avoid overflow, posterior is unnormalized anyways
        np.exp(ll, ll)
        return ll

    def extract_data(self, data=None):
        if data is None:
            data = self.data
        Dataset = namedtuple('Dataset', ['x', 'k', 'n', 'y'])
        return Dataset(data[:, 0],
                       data[:, 1]*data[:, 2],
                       data[:, 2],
                       data[:, 1])

    def jackknife_sem(self, data):
        """Run jackknife analysis on data

        This method runs a jackknife on the given data (see ml_fit for details)
        and uses this to (i) get a quick and dirty estimate of the standard
        errors that can be used during data collection to monitor the precision
        of the estimated psychometric function. (ii) detect influential
        observations. This method approach to influential observations was
        originally proposed in Wichmann & Hill (2001) and is explained in more
        detail in Fruend, Haenel, Wichmann (2011).
        """
        params = []
        for i in range(data.shape[0]):
            mask = np.ones(data.shape[0], bool)
            mask[i] = False
            params.append(self.ml_fit(data[mask],
                                      assign=False,
                                      start=self.params)[0])
        v = (len(params)-1)*np.var(np.array(params), 0)
        r = np.corrcoef(np.array(params).T)
        se = np.sqrt(v/len(params))
        self._jackknife_samples = np.array(params)
        self.sem = se
        self.param_r = r

        # Influential observations as in Fruend, Haenel, Wichmann (2011)
        influence = (np.array(params) - self.params.reshape(1, -1))/(1.96*se)
        i = np.argmax(abs(influence), 1)
        influence = np.array([infl[i_] for infl, i_ in zip(influence, i)])
        self.influence = influence

        return se, r, influence

    def deviance_test(self, data, nsamples=2000):
        raw_data = self.extract_data(data)
        p = self.predict(raw_data.x)
        k = np.random.binomial(raw_data.n.astype('i'),
                               p, size=(nsamples, len(p)))
        self.deviance_samples = deviance(k, raw_data.n, p, axis=1)
        self.deviance = deviance(raw_data.k, raw_data.n, p)
        ecdf = np.mean(self.deviance_samples <= self.deviance)
        return ecdf, self.deviance_samples

    def predict(self, x, params=None):
        """Predict psychometric function values for given x-values"""
        if params is None:
            params = self.params
        if self.guess is None:
            guess = params[0]
            params = params[1:]
        else:
            guess = self.guess
        F = self.F(self.transform(x), params[0], params[1])
        return guess + (1-guess-params[-1])*F

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
        slope, intercept = stats.linregress(self.transform(x[i]), logit)[:2]
        th = -intercept/slope
        w = 2/slope
        if self.guess is None:
            return np.array([th, w, 0.02, 0.02])
        else:
            return np.array([th, w, 0.02])


def deviance(k, n, p, axis=0, clip=1e-7):
    p0 = np.clip(k/n, clip, 1-clip)
    lfit = np.sum(k*np.log(p) + (n-k)*np.log1p(-p), axis=axis)
    lsat = np.sum(k*np.log(p0) + (n-k)*np.log1p(-p0), axis=axis)
    return 2*(lsat - lfit)


def bayesian_inference(pmf,
                       statistics={'threshold': lambda params: params[0],
                                   'width': lambda params: params[1]},
                       nexamples=0):
    """Perform bayesian inference using direct integration of the posterior

    Args:
        pmf:
            Psychometricfunction object
        statistics:
            a dictionary of the statistics to be determined. Each key is a
            label for the respective statistic, each value should be a function
            that takes a parameter vector and returns the desired statistic.
            The default just gets the parameters of the psychometric function,
            but more elaborate functions could derive more complicated
            characteristics from the parameters.
        nexamples:
            integer number of posterior examples to be returned. If this is 0,
            there will be no third return value.

    Returns:
        posterior_stats, (examples)
        posterior_stats:
            Dictionary of posterior mean and posterior variance for each
            statistic. Keys are the same as for the input argument `statistics`
        examples:
            if posterior examples are required, they are returned as a second
            argument.
    """
    grid = mkgrid(
        *[(par - 3*se, par + 3*se, 20) for par, se in zip(pmf.params, pmf.sem)]
    )
    grid, posterior = integrate_posterior(pmf, grid)
    if nexamples:
        idx = sample_gridpoints(nexamples, posterior)
        return ({key: get_stats(posterior, getter(grid))
                 for key, getter in statistics.items()},
                zip(grid[:, idx].T, posterior[idx]))
    else:
        return {key: get_stats(posterior, getter(grid))
                for key, getter in statistics.items()}


def sample_gridpoints(nexamples, posterior):
    idx = np.random.multinomial(1, posterior, size=nexamples)
    idx = np.where(idx > 0)
    idx = idx[1]
    return idx


def pmfplot(pmf, examples=None, **kwargs):
    """Draw a canonic psychometric function plot

    Args:
        pmf:
            fitted psychometric function model
        draw_scatter:
            Either of the following
                - 'none': No data points are shown
                - 'dots': Show individual block data as dots in the plot
                - 'influence': Overlay influence numbers from Fruend, Haenel,
                    Wichmann (2011)
        axes:
            matplotlib axes to use for the plot
    """
    if 'axes' in kwargs:
        ax = kwargs.pop('axes')
    else:
        if HASPYLAB:
            ax = pl.axes()
        else:
            raise RuntimeError(
                'Trying to use pylab function by pylab does not exist')
    if 'draw_scatter' in kwargs:
        draw_scatter = kwargs.pop('draw_scatter')
    else:
        draw_scatter = 'dots'

    col = kwargs.setdefault('color', 'black')

    dataset = pmf.extract_data()
    xmin = dataset.x.min()
    xmax = dataset.x.max()
    x = kwargs.pop('x', np.mgrid[xmin:xmax:100j])
    ibelow = x <= xmin
    iabove = x >= xmax
    idatarange = np.logical_and(x >= xmin, x <= xmax)

    p = pmf.predict(x)
    if examples is None:
        if pmf.sem is not None:
            se = determine_error_region(pmf, x)
            ax.fill_between(x, p+se, p-se, facecolor=col, alpha=0.5)
    else:
        for ex in examples:
            p = pmf.predict(x, ex[0])
            plot_single_pmf(ax, x, p, ibelow, iabove, idatarange,
                            alpha=400*ex[1], **kwargs)

    plot_single_pmf(ax, x, p, ibelow, iabove, idatarange,
                    **kwargs)

    if draw_scatter in ['dots', 'influence']:
        ax.scatter(dataset.x, dataset.y, s=dataset.n, c=col)
        if draw_scatter == 'influence':
            for x, y, i in zip(dataset.x, dataset.y, pmf.influence):
                pl.text(x, y, '{:.2f}'.format(i), horizontalalignment='center')


def plot_single_pmf(ax, x, p, ibelow, iabove, idatarange, **kwargs):
    col = kwargs['color']
    ax.plot(x[ibelow], p[ibelow], '--', color=col)
    ax.plot(x[iabove], p[iabove], '--', color=col)
    ax.plot(x[idatarange], p[idatarange], **kwargs)


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
