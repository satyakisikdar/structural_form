'''
structcounts.m
params.logps{i}(n) is prior for an i-structure with n clusters
we compute and store the priors in advance
'''
import numpy as np
from math import factorial
from scipy.special import gammaln

def stirling2(n, m):
    s2 = np.zeros((n, m))
    s2[0, 0] = 1
    s2[0, 1:] = 0

    for i in range(1, n):
        s2[i, 0] = 1
        for j in range(1, m):
            s2[i, j] = (j + 1) * s2[i-1, j] + s2[i-1, j-1]
    return s2

def sum_logs(x):
    mx = np.max(x)
    xp = np.exp(x - mx)
    return np.log(np.sum(xp)) + mx

def struct_counts(params, num_objects):
    max_n = num_objects
    theta = params.theta  # 1 - np.exp(-3)
    s2_mat = stirling2(max_n, max_n)

    F = np.array([factorial(i + 1) for i in range(max_n)])

    params.T = np.tile(F, (max_n, 1)) * s2_mat  # dunno what this is

    log_counts = np.zeros((8, max_n))  #  log_counts(i, n) is log #architectures of type i with n labelled clustersrows correspond to structures, columns correspond to n

    # partitions
    log_counts[0, :] = np.zeros((1, max_n))

    # ring
    log_counts[1, :] = gammaln(np.arange(1, max_n+1) + 1) - np.log(2)
    log_counts[1, 0] = 0

    # ring
    log_counts[2, :] = gammaln(np.arange(1, max_n+1)) - np.log(2)
    log_counts[2, : 2] = (0, 0)

    # unrooted tree
    log_counts[3, 1:] = gammaln(np.arange(2, max_n+1) - 1.5) + (np.arange(2, max_n+1) - 2) * np.log(2) - 0.5 * np.log(np.pi)
    log_counts[3, 0] = 0

    # unrooted hierarchy
    log_counts[4, :] = (np.arange(1, max_n+1) - 2) * np.log(np.arange(1, max_n+1))

    # rooted hierarchy
    log_counts[5, :] = (np.arange(1, max_n+1) - 1) * np.log(np.arange(1, max_n+1))

    # directed chain
    log_counts[6, :] = gammaln(np.arange(1, max_n+1) + 1)

    # directed ring
    log_counts[7, :] = gammaln(np.arange(1, max_n+1))

    # we choose among all structures where each dimension contains no holes (but
    # multiple objects can end up at the same node). Each structure is weighted
    # according to the number of nodes it contains.

    # consider the number of ways to partition the objects into each ncluster
    log_cluster_counts = np.log(s2_mat[max_n - 1, :])

    # combine npartitions with narchitectures to get number of structures
    log_counts = log_counts + np.tile(log_cluster_counts, (8, 1))

    log_weights = np.log(theta) + np.arange(1, max_n+1) * np.log(1 - theta)


    tot_sums = np.array([sum_logs(log_weights + log_counts[i, :]) for i in range(8)])

    params.log_priors = np.tile(log_weights, (8, 1)) - np.tile(np.transpose(tot_sums), (1, max_n))
    return params
