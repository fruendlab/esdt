# Extended Signal Detection Theory with Decision Value Correlation

This python module implements some basic methods for analyzing
psychophysical data. In particular, it implements decision value
correlation as described in the paper

Sebastian & Geisler (2018): Decision-Variable Correlation. J Vis, 18(4):
3.

## Install

You can use
```
pip install esdt
```
to install the package.

The package is a [pybuilder](http://pybuilder.github.io/) project. Check
out their documentation for how to build and install it.

## Basic usage

There are two modules `esdt.sdt` for signal detection theory (and decision
variable correlation) and `esdt.pmf` for psychometric function estimation.

## Fitting a psychometric function

Create a 2AFC psychometric function with logistic sigmoid.

```python
>>> import numpy as np
>>> from esdt import pmf
>>> psychometric = pmf.PsychometricFunction(pmf.logistic, 0.5)
>>> data = np.array([[0.1, 0.4, 5],
...                  [0.2, 0.4, 5],
...                  [0.3, 0.6, 5],
...                  [0.4, 0.8, 5],
...                  [0.6, 1.0, 5]])
>>> params, ll = psychometric.ml_fit(data)

```

Note that the data format has stimulus intensity in the first column, then
the fraction of correct responses and then the number of responses
collected.

While we collect data, it is often useful to have a quick and dirty way to
assess how the fitted function looks in order to decide were we need more
samples. We implemented a jackknife procedure to get "quick-and-dirty"
standard errors:

```python
>>> se, r, infl = psychometric.jackknife_sem(data)

```

This will give you a very rough idea of the standard error and of
correlations between parameters. Furthermore, the third output is
a measure for the influence of the individual blocks in your data
structure. Note that standard errors based on jackknife should *not* be
used as final estimates. In general, it seems that Bayesian inference
tends to give more honest estimates of confidence for parameters of the
psychometric function.

In order to refine your analysis using Bayesian inference, you can now
take the psychometric function model and determine Bayesian confidence
intervals for a number of parameters of interest:

```python
>>> ci = pmf.bayesian_inference(psychometric)
>>> print(ci)
{'threshold': (0.4..., 0.0...), 'width': ...}

```

Also see the examples in the integration tests folder for more details.
