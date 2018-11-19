"""
This file provides a basic example on using the pmf module
"""
import pylab as pl
import numpy as np

from esdt import pmf

# Here is some simple raw data
# The first column is stimulus intensity
# The second column is the fraction of correct responses
# The third column is the total number of responses collected
data = [[.3, .5, 20],
        [.5, .6, 20],
        [.7, .85, 20],
        [.9, .95, 20],
        [1.1, 1., 20]]
data = np.array(data)

# Create a psychometric function object. Here we're using a logistic sigmoid
# and we assume that we know the guessing rate (i.e. lower asymptote) to be
# 0.5. This would for example be the case for a psychometric function from a 2
# alternative forced choice paradigm.
pf = pmf.PsychometricFunction(pmf.logistic, 0.5)

# The psychometric function now contains our model specification, but it hasn't
# made contact with data yet. The ml_fit method performs a maximum likelihood
# fit.
pf.ml_fit(data)

# A quick and dirty way to get confidence regions is the jackknife. It is great
# to get an idea if your errors are in a reasonable ballpark, but it will
# generally *underestimate* the errorbars. So don't use this for any real
# analysis, but rather use the Bayesian inference approach for data analysis,
# that you want to publish.
se, r, infl = pf.jackknife_sem(data)

# As you can see, the jackknife procedure returns three measures,
# 1. a coarse standard error for each of your parameters (se)
# 2. an estimate of the correlation between different parameters (r)
# 3. an estimate of the influence of each individual data point on the fit
#    (infl). The influence measure was originally developed by Wichmann & Hill
#    (2001) and has been described and studied in more detail by Fruend et al
#    (2011).

# Next we might want to plot a psychometric function. Here, we use the
# `draw_scatter` argument, to indicate that the scatterplot of our observed
# data should show numbers to indicate the respective influence values.
pmf.pmfplot(pf, draw_scatter='influence')

pl.show()
