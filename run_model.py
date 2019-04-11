'''
MATLAB file: runmodel.m
'''
import os
import networkx as nx
import numpy as np
from copy import deepcopy
from typing import Tuple, List, Set, Any


from utilities import load_from_mat
from graph_priors import set_log_priors, graph_prior
from graph_likelihoods import graph_likelihood
from Forms import ClusterGraph
from parameters import Parameters


class Record:
    def __init__(self, cnode: int, log_likelihood: float, parts: Tuple[Set[int], Set[int]], cluster_graph: ClusterGraph) -> None:
        self.cnode = cnode
        self.log_likelihood = log_likelihood
        self.parts = parts
        self.cluster_graph = cluster_graph


def calculate_log_likelihood(cluster_graph: ClusterGraph, data_graph: nx.DiGraph, params: Parameters) -> Tuple[float, ClusterGraph]:
    '''
    :param cluster_graph:
    :param data_graph:
    :param params:
    :return:
    '''
    log_likelihood, cluster_graph = graph_likelihood(data_graph, cluster_graph, params)
    log_likelihood += graph_prior(cluster_graph, params)
    return log_likelihood, cluster_graph


def choose_node_split(cnode: int, cluster_graph: ClusterGraph, params: Parameters) -> Tuple[float, Set[int], Set[int]]:
    partition_members = cluster_graph.node[cnode]['members']
    print(f'Splitting cluster id {cnode}')

    if len(partition_members) == 1:
        log_likelihood = np.float('-inf')
        part_1 = {cnode}
        part_2 = set()
    else:
        _, log_likelihood, part_1, part_2 = cluster_graph.get_best_split(cnode, params)

    return log_likelihood, part_1, part_2


def optimize_depth(cluster_graph: ClusterGraph, depth: int, log_likelihoods: float, new_cluster_graph: nx.DiGraph,
                   data_graph: nx.DiGraph, params: Parameters) -> None:
     for cnode in cluster_graph.nodes():
         if new_cluster_graph.number_of_nodes() == 0:
             continue
         log_likelihoods, new_cluster_graph = calculate_log_likelihood(new_cluster_graph, data_graph, params)
     return log_likelihoods, new_cluster_graph


def structure_fit(data_graph: nx.DiGraph, params: Parameters, cluster_graph: ClusterGraph) -> Tuple[float, ClusterGraph, List[float],
                                                                                            List[nx.DiGraph]]:
    # TODO first split matches... likelihood numbers are off for the second iteration and on  - data_graph subgraph is turned off..
    loop_eps = 10 ** (-2)

    best_graph_log_likelihoods = []
    best_graphs = []

    current_probab, cluster_graph = calculate_log_likelihood(cluster_graph, data_graph, params)

    stop_flag = False
    depth = 0

    records = []
    # split cluster nodes while score improves
    # while not stop_flag:
    for _ in range(2):
        best_record_at_depth = None
        max_log_likehood = float('-inf')
        records_at_depth = []
        cluster_graph_copy = deepcopy(cluster_graph)

        for cnode in cluster_graph_copy.nodes():
            cluster_graph_copy, log_likelihood, part_1, part_2 = cluster_graph_copy.get_best_split(cnode, params=params)  # figure out what happens here, I think cluster_graph.split_node would work

            # cluster_graph_copy.split_node(cnode, (part_1, part_2))

            record_at_depth = Record(cnode=cnode, log_likelihood=log_likelihood, parts=(part_1, part_2),
                                     cluster_graph=cluster_graph_copy)

            records_at_depth.append(record_at_depth)

            if record_at_depth.log_likelihood > max_log_likehood:
                max_log_likehood = record_at_depth.log_likelihood
                best_record_at_depth = record_at_depth

        records.append(records_at_depth)

        if max_log_likehood == float('-inf'):  # no splits possible
            # update log_likelihoods and new_graph
            best_record_at_depth.log_likelihood = current_probab
            best_record_at_depth.cluster_graph = cluster_graph_copy

        new_score = best_record_at_depth.log_likelihood
        new_cluster_graph = best_record_at_depth.cluster_graph

        # new_score, new_cluster_graph = gibbs_clean()  # TODO for laters
        # new_score, new_cluster_graph = gibbs_clean()  # TODO for laters

        if depth < 10 or new_score - current_probab <= loop_eps:
            print('small depth: optimizing branch lengths of best split')
            new_score, new_cluster_graph = calculate_log_likelihood(new_cluster_graph, data_graph, params)

        if new_score - current_probab <= loop_eps:
            print('do slow gibbs clean - NOT IMPLEMENTED yet')
            # TODO new_score, new_cluster_graph = gibbs_clean()

            # optimize all splits at this depth
            print('optimize all splits at current depth')
            new_score, new_cluster_graph = calculate_log_likelihood(new_cluster_graph, data_graph, params)
            try_new_cluster_graph, try_log_likelihood, _, _ = try_new_cluster_graph.get_best_split(depth, params)

            # go through, see what's redundant and not. get rid of flags change
            # then recompute
            # brlencases. feel free to burn
            # we don't touch any flags in the likelihood
            # stretch goal: look at choose seedpairs, implemen=
            # on thursday: go through simplify_graph.m

            if try_log_likelihood > new_score: # without gibbs clean, this is redundant
                new_score = log_likelihoods
                new_cluster_graph = try_new_cluster_graph

        if new_score - current_probab <= loop_eps:
            stop_flag = True   # the score cannot be beaten

        else:
            print('Improvement:', new_score - current_probab)
            best_record_at_depth.cluster_graph = new_cluster_graph
            best_record_at_depth.log_likelihood = new_score

            cluster_graph = new_cluster_graph
            current_probab = new_score

            best_graphs.append(cluster_graph)
            best_graph_log_likelihoods.append(current_probab)

            depth += 1
    return current_probab, cluster_graph, best_graph_log_likelihoods, best_graphs


def simplify_graph(data_graph: nx.DiGraph, params: Parameters):
    # remove:
    # 1. dangling cluster nodes: any node that's not an object node but has 0-1 cluster neighbors, no
    #    object neighbors
    # 2. any cluster node with exactly two neighbors, one of which is a cluster node

    # TODO: why cont = ones(1, 3)?
    # TODO: need call to combine_graphs, redundant_inds
    pass


def run_model(struct_index: int, data_index: int) -> \
        Tuple[float, nx.DiGraph, List[float], List[nx.DiGraph]]:
    '''
    Given data set DIND, find the best instance of form SIND.
    :param params: DefaultParameters object
    :param struct_index: index to structure
    :param data_index: index to dataset
    :param repeat_index: which repeat is this?
    :param save_file: where to save interim results
    :return:
        log_probability: log probability of the best structure found
        graph: the best substructure found
        names: TBD
        best_graph_log_probs: log probabilities of structures explored along the way
        best_graph: structures along the way
    '''

    data_graph = nx.Graph()
    # data_g.add_edges_from([(1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 6), (7, 8), (7, 9), (8, 9), (10, 11), (10, 12),
    #                         (11, 12), (3, 5), (6, 8), (9, 11)])
    data_graph.add_edges_from([(1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 6), (3, 5)])
    data_graph.name = 'prisoners'
    data_graph = nx.DiGraph(data_graph)

    # read data_graph from file
    # data_graph = nx.DiGraph()
    cluster_graph = ClusterGraph(struct_type='chain', dataset='test', data_graph=data_graph)

    params = Parameters()
    params.set_runtime_parameters(data_graph=data_graph, struct_name=cluster_graph.struct_type)

    return structure_fit( data_graph=data_graph, params=params, cluster_graph=cluster_graph, best_graph_log_likelihoods=[],
        best_cluster_graphs=[])


def main():
    run_model(0, 0)
    # data_g = nx.Graph()
    # # data_g.add_edges_from([(1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 6), (7, 8), (7, 9), (8, 9), (10, 11), (10, 12),
    # #                         (11, 12), (3, 5), (6, 8), (9, 11)])
    # data_g.add_edges_from([(1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 6), (3, 5)])
    # data_g.name = 'prisoners'
    # data_g = nx.DiGraph(data_g)
    #
    # cluster_graph = ClusterGraph(struct_type='chain', dataset='test', data_graph=data_g)
    #
    # params = Parameters()
    # params.set_runtime_parameters(data_graph=data_g, struct_name=cluster_graph.struct_type)
    #
    # _ = structure_fit(data_g, params, cluster_graph)
    # print()


if __name__ == '__main__':
    main()
