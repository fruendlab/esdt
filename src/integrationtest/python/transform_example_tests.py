import numpy as np
from esdt import pmf


data = np.array([
    [0.02, 0.5, 20],
    [0.025, 0.6, 20],
    [0.03, 0.55, 20],
    [0.04, 0.6, 20],
    [0.05, 0.65, 20],
    [0.065, 0.8, 20],
    [0.08, 0.7, 20],
    [0.1, 0.6, 20],
    [0.15, 0.8, 20],
    [0.2, 0.6, 20],
    [0.25, 0.75, 20],
    [0.275, 0.8, 20],
    [0.325, 0.9, 20],
    [0.325, 0.85, 20],
    [0.35, 0.8, 20],
    [0.35, 0.9, 20],
    [0.4, 0.8, 20],
    [0.4, 0.95, 20],
    [0.45, 0.95, 20],
    [0.5, 0.95, 20],
    [0.55, 0.7, 20]])

data_log = data.copy()
data_log[:, 0] = np.log10(data[:, 0])

ps_log = pmf.PsychometricFunction(pmf.gumbel, 0.5)
ps_trans = pmf.PsychometricFunction(pmf.gumbel, 0.5, transform=np.log10)

params_log, ll_log = ps_log.fit(data_log)
params_trans, ll_trans = ps_trans.fit(data)

print(params_log, params_trans)
print(ll_log, ll_trans)

l0 = ps_log.negloglikelihood(params_log,
                             data_log[:, 0],
                             data_log[:, 1],
                             data_log[:, 2])

l1 = ps_trans.negloglikelihood(params_log,
                               data[:, 0],
                               data[:, 1],
                               data[:, 2])

assert l0 == l1
assert ll_log == ll_trans
assert abs(params_log - params_trans).max() < 1e-6
