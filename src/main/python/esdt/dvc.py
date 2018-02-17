from collections import namedtuple
import numpy as np
from scipy import stats
from scipy import optimize


ResponseCollection = namedtuple('ResponseCollection',
                                ['pAA', 'pAB', 'pBA', 'pBB'])


def dvc_for_category(responses_1, responses_2, dprimes, criteria):
    responses_1 = np.asarray(responses_1)
    responses_2 = np.asarray(responses_2)
    responses = ResponseCollection(
        np.logical_and(responses_1 == 1, responses_2 == 1).mean(),
        np.logical_and(responses_1 == 1, responses_2 == 0).mean(),
        np.logical_and(responses_1 == 0, responses_2 == 1).mean(),
        np.logical_and(responses_1 == 0, responses_2 == 0).mean(),
    )

    return dvc_from_probs(responses, dprimes, criteria)


def dvc_from_probs(responses, dprimes, criteria):
    rho = optimize.fmin(loss, [0.], args=(responses, dprimes, criteria))
    return rho


def loss(rho, responses, dprimes, criteria):
    infinity = 100
    means = 0.5*dprimes
    Sigma = np.eye(2)
    Sigma[0, 1] = Sigma[1, 0] = rho
    fAA, _ = stats.mvn.mvnun([-infinity, -infinity], criteria, means, Sigma)
    fAB, _ = stats.mvn.mvnun([criteria[0], -infinity], [infinity, criteria[1]],
                             means, Sigma)
    fBA, _ = stats.mvn.mvnun([-infinity, criteria[1]], [criteria[0], infinity],
                             means, Sigma)
    fBB, _ = stats.mvn.mvnun(criteria, [infinity, infinity], means, Sigma)
    return - np.dot(np.asarray(responses),
                    np.log(np.array([fAA, fAB, fBA, fBB])))
