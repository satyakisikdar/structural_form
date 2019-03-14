'''
graph_like_rel.m
'''
import numpy as np
import numpy.matlib
import networkx as nx
from scipy.special import gammaln
from collections import defaultdict

def get_alpha_beta_pair(n):
    '''
    return n pairs of (alpha_0, beta_0) and (alpha_1, beta_1)
    '''
    np.random.seed(123)
    a = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16, 32]
    b = np.arange(0.05, 1, 0.05)
    pairs_0, pairs_1 = [], []

    while len(pairs_0) != n and len(pairs_1) != n:
        sum_alpha_beta_0 = a[np.random.choice(np.arange(len(a)))]
        alpha_ratio_0 = np.random.choice(b)

        sum_alpha_beta_1 = a[np.random.choice(np.arange(len(a)))]
        alpha_ratio_1 = np.random.choice(b)

        if alpha_ratio_1 >= alpha_ratio_0:
            alpha_0 = alpha_ratio_0 * sum_alpha_beta_0
            beta_0 = sum_alpha_beta_0 - alpha_0
            pairs_0.append((round(alpha_0, 3), round(beta_0, 3)))

            alpha_1 = alpha_ratio_1 * sum_alpha_beta_1
            beta_1 = sum_alpha_beta_1 - alpha_1
            pairs_1.append((round(alpha_1, 3), round(beta_1, 3)))

    return pairs_0[: n], pairs_1[: n]


def get_pairwise_cluster_counts(cluster_graph, data_graph):
    # counts = defaultdict(int)
    counts = np.zeros((cluster_graph.order(), cluster_graph.order()))

    # re-index cluster labels to start from 0
    reindexed_cluster_labels = {}
    start = 0
    for cluster in cluster_graph.cluster_labels.values():
        if cluster not in reindexed_cluster_labels:
            reindexed_cluster_labels[cluster] = start
            start += 1

    num_objects = cluster_graph.num_objects
    # counts[('0', '0')] = (cluster_graph.order() + data_graph.order()) - num_objects
    all_members = set()
    for node, data in cluster_graph.nodes(data=True):
        # print(node, data)
        all_members.update(data['members'])

    for cluster_node in cluster_graph.nodes():
        # each cluster node has a members list
        members = cluster_graph.node[cluster_node]['members']
        for member in members:
            # each member correspond to a node in the data graph
            for neighbor in data_graph.neighbors(member):
                if neighbor not in all_members:  # don't consider nodes not present in the cluster_graph
                    continue
                member_cluster = reindexed_cluster_labels[cluster_graph.cluster_labels[member]]
                neighbor_cluster = reindexed_cluster_labels[cluster_graph.cluster_labels[neighbor]]

                counts[neighbor_cluster, member_cluster] += 1

    return counts


def trans2orig(Ps, Ss):
    alphas = Ps * Ss
    betas = Ss - alphas
    return alphas, betas


def make_hyperparams(props, sums):
    '''
    make a grid of params given proportions PROPS and sums SUMS
    :param props:
    :param sums:
    :return:
    '''
    Ps, Ss = np.meshgrid(props, sums)
    alphas, betas = trans2orig(Ps, Ss)
    return alphas, betas


def beta_binomial_log_likelihood(alpha, beta, ns, ys):
    log_likelihood = np.sum(gammaln(alpha + ys) + gammaln(beta + (ns-ys)) - gammaln(ns + alpha + beta) - gammaln(alpha)
                            - gammaln(beta) + gammaln(alpha + beta), axis=0)
    return log_likelihood


def mean_logs(X):
    X = X[np.isfinite(X)]
    mx = np.nanmax(np.nanmax(X))
    if ~np.isfinite(mx):
        return float('-inf')
    Xp = np.exp(X - mx)
    # print('mean log:', np.round(np.mean(Xp), 3), np.round(np.nanmean(Xp), 3), 'max', mx)
    mean_Xp = np.nanmean(Xp)
    if mean_Xp != 0:
        L = np.log(np.nanmean(Xp)) + mx
    else:
        L = float('-inf')
    return L


def _graph_likelihood(count_vec, adj_vec, size_vec, mags, thetas):
    '''
    works
    rellikebin.m
    :param count_vec:
    :param cluster_graph:
    :param size_vec:
    :param mags:
    :param thetas:
    :return:
    '''
    mags_mat = np.matlib.repmat(mags, 1, thetas.size)
    thetas_mat = np.matlib.repmat(np.transpose(thetas), mags.size, 1)
    alphas_mat = np.multiply(thetas_mat, mags_mat)
    betas_mat = mags_mat - alphas_mat

    alphas = np.array([alphas_mat.flatten(order='F')])  # flatten column-wise aka Fortran style
    betas = np.array([betas_mat.flatten(order='F')])
    hyp_count = alphas.size

    ## TODO: figure out for loop in line 19 in rellikebin
    vals = [0, 1]

    # use numpy.where to find the indices
    log_likelihood_mat = {}
    for v in vals:
        ys = np.array([count_vec[adj_vec==v]])
        ns = np.array([size_vec[adj_vec==v]])
        d_count = ys.size

        if len(ys) == 0:
            log_likelihood_mat[v] = np.zeros_like(mags_mat)
        else:
            all_ys = np.repeat(np.transpose(ys), hyp_count, axis=1)
            all_ns = np.repeat(np.transpose(ns), hyp_count, axis=1)

            all_alphas = np.repeat(alphas, d_count, axis=0)
            all_betas = np.repeat(betas, d_count, axis=0)

            log_likelihoods = beta_binomial_log_likelihood(all_alphas, all_betas, all_ns, all_ys)
            log_likelihood_mat[v] = log_likelihoods.reshape((alphas_mat.shape[1], alphas_mat.shape[1]), order='F')  # Fortran style, because MATLAB

    zeros = np.zeros(log_likelihood_mat[0].shape[1])
    ones = np.zeros(log_likelihood_mat[0].shape[1])

    for i in range(log_likelihood_mat[0].shape[1]):
        zeros[i] = mean_logs(log_likelihood_mat[0][:, i])
        ones[i] = mean_logs(log_likelihood_mat[1][:, i])

    zero_idx, one_idx = np.triu_indices(zeros.shape[0])
    log_likelihoods = zeros[zero_idx] + ones[one_idx]
    log_likelihood = mean_logs(log_likelihoods)
    return log_likelihood


def graph_likelihood(data_graph, cluster_graph, params):
    '''
    works
    graph_like_rel.m

    computes log P(data | cluster_graph)
    :param data:
    :param cluster_graph:
    :param params:
    :return:
    '''
    original_cluster_graph = cluster_graph

    current_objects, class_labels = zip(*cluster_graph.cluster_labels.items())

    data_graph = data_graph.subgraph(current_objects)

    if cluster_graph.struct_type.startswith('undir'):
        cluster_graph.to_undirected()

    num_clusters = cluster_graph.order()

    class_sizes, _ = np.histogram(class_labels, bins=num_clusters)  # potential bug
    class_sizes = np.array([class_sizes])
    size_matrix = np.multiply(np.matlib.repmat(class_sizes, num_clusters, 1),
                              np.matlib.repmat(np.transpose(class_sizes), 1, num_clusters))

    for i in range(size_matrix.shape[0]):
        size_matrix[i, i] -= class_sizes[0, i]

    class_counts = get_pairwise_cluster_counts(cluster_graph, data_graph)
    count_vec = class_counts.flatten()  # make it into a 2-D
    adj_vec = np.array(nx.to_numpy_matrix(cluster_graph)).flatten()  # np.array converts matrix to an array which can be flattened
    size_vec = size_matrix.flatten()


    # edges in data_graph should encourage links in cluster_graph - the following lines are not used for binary data
    # edge_prop_steps = 5
    # edge_sum_steps = params.edge_sum_steps
    # edge_offset = params.edge_offset
    # edge_sums = np.power(params.edge_sum_lambda, np.arange(params.edge_offset + 1, edge_offset + edge_sum_steps + 1,
    #                                                        dtype=float))
    # edge_props = np.arange(edge_prop_steps + 1, 2 * edge_prop_steps + 1) / (2 * edge_prop_steps) - (1 / (4 * edge_prop_steps))
    # no_edge_props = np.arange(1, edge_prop_steps + 1) / (2 * edge_prop_steps) - (1/(4 * edge_prop_steps))

    # alphas_edge, betas_edge = make_hyperparams(edge_props, edge_sums)
    # alphas_no_edge, betas_no_edge = make_hyperparams(no_edge_props, edge_sums)

    # for binary data
    edge_prop_steps = 10
    edge_sum_steps = params.edge_sum_steps
    edge_offset = params.edge_offset
    mags = np.array([np.power(params.edge_sum_lambda, np.arange(edge_offset + 1, edge_offset + edge_sum_steps + 1, dtype=float))])
    thetas = np.array([np.arange(1, edge_prop_steps + 1) / edge_prop_steps - (1 / (2 * edge_prop_steps))])

    log_likelihood = _graph_likelihood(count_vec, adj_vec, size_vec, np.transpose(mags), np.transpose(thetas))

    return log_likelihood, original_cluster_graph