import numpy as np
from scipy import stats


def basic_sdt_analysis(targets, responses):
    hits = np.asarray(responses)[np.asarray(targets, bool)]
    fa = np.asarray(responses)[np.logical_not(np.asarray(targets, bool))]
    return basic_sdt(hits.mean(), fa.mean())


def basic_sdt(hit_rate, false_alarm_rate):
    dprime = stats.norm.ppf(hit_rate) - stats.norm.ppf(false_alarm_rate)
    gamma = -0.5*(stats.norm(-0.5*dprime).ppf(false_alarm_rate) +
                  stats.norm(0.5*dprime).ppf(hit_rate))
    return dprime, gamma
