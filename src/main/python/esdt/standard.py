from scipy import stats


def basic_sdt_analysis(targets, responses):
    hits = responses[targets].mean()
    fa = responses[1-targets].mean()
    return basic_sdt(hits, fa)


def basic_sdt(hit_rate, false_alarm_rate):
    dprime = stats.norm.ppf(hit_rate) - stats.norm.ppf(false_alarm_rate)
    gamma = -0.5*(stats.norm(-0.5*dprime).ppf(false_alarm_rate) +
                  stats.norm(0.5*dprime).ppf(hit_rate))
    return dprime, gamma
