"""
This file provides an example on using the bayesian inference function
of the pmf module. It is an extension of the basic_example.py that also
contains more data about the general setup
"""
import pylab as pl
import numpy as np
from scipy import stats

from esdt import pmf

# Raw data
data = [[.3, .5, 20],
        [.5, .6, 20],
        [.7, .85, 20],
        [.9, .95, 20],
        [1.1, 1., 20]]
data = np.array(data)

# Create a psychometric function
pf = pmf.PsychometricFunction(pmf.logistic, 0.5)

# Now we want to set priors on the psychometric function object to make sure
# we can do Bayesian inference. These priors should encode prior knowledge that
# we have about "typical" psychometric functions on datasets like ours. In many
# cases, we can assume that we sampled the psychometric function reasonably
# well so that priors can be derived from the data.
pf.priors = (
    stats.uniform(.3, 1.1),  # Threshold (expected within data range)
    stats.uniform(.2, .8),  # Slope (between smallest and largest difference)
    stats.beta(2, 20),  # Guessing rate is often well captured by this prior
)

# We now perform an ml_fit and the jackknife to get an idea where our
# psychometric function is.
pf.ml_fit(data)
pf.jackknife_sem(data)

# This actually performs Bayesian inference by numerically integrating the
# posterior on a grid. Sometimes it is nice for plotting to show a couple of
# posterior samples. Setting `nexamples` asks for a couple of such samples that
# can then be passed to the plot function.
est, examples = pmf.bayesian_inference(pf, nexamples=50)

# Now est contains a dictionary in which every key corresponds to one
# descriptor of the psychometric function and the following two values are the
# mean and standard deviation of the posterior for that descriptor. Note that
# the `statistics` argument of `bayesian_inference` allows you to compute
# arbitrary descriptors from the parameters of the psychometric function and
# est will contain mean and standard deviation for each one of them.
print(est)

# Plot pmf
pmf.pmfplot(pf, examples)
pl.show()
